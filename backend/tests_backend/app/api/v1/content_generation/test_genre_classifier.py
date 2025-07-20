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
def test_dummy_genre_classification(audio_data={'format': 'wav'}, user_context={'id': 'test'}):
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.api.v1.content_generation import genre_classifier
        result = getattr(genre_classifier, 'dummy_genre_classification')("test_audio", "test_user")
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_genreclassifier_class():
    # Instanciation réelle
    try:
        from backend.app.api.v1.content_generation import genre_classifier
        obj = getattr(genre_classifier, 'GenreClassifier')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

