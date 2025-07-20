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
def test_save_file():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import file_utils
        result = getattr(file_utils, 'save_file')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_remove_file():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import file_utils
        result = getattr(file_utils, 'remove_file')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_file_size():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import file_utils
        result = getattr(file_utils, 'file_size')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_is_allowed_extension():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.utils import file_utils
        result = getattr(file_utils, 'is_allowed_extension')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

