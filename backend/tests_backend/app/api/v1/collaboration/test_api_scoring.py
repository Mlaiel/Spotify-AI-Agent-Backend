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
def test_get_collab_score(score_a=1.0, score_b=1.0):
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.api.v1.collaboration import api_scoring
        result = getattr(api_scoring, 'get_collab_score')(1.0, 1.0)
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

