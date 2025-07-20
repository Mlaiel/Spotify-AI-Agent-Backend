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
def test_setup_logging():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.core.logging import logger_config
        result = getattr(logger_config, 'setup_logging')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

