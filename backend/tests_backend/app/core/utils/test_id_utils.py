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
def test_generate_uuid():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import id_utils
        result = getattr(id_utils, 'generate_uuid')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_generate_short_id():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import id_utils
        result = getattr(id_utils, 'generate_short_id')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

