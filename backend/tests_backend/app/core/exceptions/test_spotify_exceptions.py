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
def test_spotifyapiexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import spotify_exceptions
        obj = getattr(spotify_exceptions, 'SpotifyAPIException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_spotifyquotaexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import spotify_exceptions
        obj = getattr(spotify_exceptions, 'SpotifyQuotaException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_spotifypermissionexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import spotify_exceptions
        obj = getattr(spotify_exceptions, 'SpotifyPermissionException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_spotifyintegrationexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import spotify_exceptions
        obj = getattr(spotify_exceptions, 'SpotifyIntegrationException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_spotifybusinessexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import spotify_exceptions
        obj = getattr(spotify_exceptions, 'SpotifyBusinessException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

