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
def test_get_plural_rule():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.i18n.utils import pluralization
        result = getattr(pluralization, 'get_plural_rule')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_pluralize():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.i18n.utils import pluralization
        result = getattr(pluralization, 'pluralize')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

