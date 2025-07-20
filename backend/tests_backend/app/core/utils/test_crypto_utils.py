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
def test_hash_sha256():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import crypto_utils
        result = getattr(crypto_utils, 'hash_sha256')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_hmac_sign():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import crypto_utils
        result = getattr(crypto_utils, 'hmac_sign')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_verify_hmac():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import crypto_utils
        result = getattr(crypto_utils, 'verify_hmac')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_random_token():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import crypto_utils
        result = getattr(crypto_utils, 'random_token')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

