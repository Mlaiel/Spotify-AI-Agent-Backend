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
def test_spotifydataresponse_class():
    # Instanciation réelle
    try:
        from backend.app.schemas.response import spotify_response
        obj = getattr(spotify_response, 'SpotifyDataResponse')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_spotifydatadeleteresponse_class():
    # Instanciation réelle
    try:
        from backend.app.schemas.response import spotify_response
        obj = getattr(spotify_response, 'SpotifyDataDeleteResponse')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

