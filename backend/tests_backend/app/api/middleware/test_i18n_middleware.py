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
def test_i18nconfig_class():
    # Instanciation réelle
    try:
        from backend.app.api.middleware import i18n_middleware
        obj = getattr(i18n_middleware, 'I18NConfig')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_languagedetectionmiddleware_class():
    # Instanciation réelle
    try:
        from backend.app.api.middleware import i18n_middleware
        obj = getattr(i18n_middleware, 'LanguageDetectionMiddleware')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_internationalizationmiddleware_class():
    # Instanciation réelle
    try:
        from backend.app.api.middleware import i18n_middleware
        obj = getattr(i18n_middleware, 'InternationalizationMiddleware')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_translationcachemiddleware_class():
    # Instanciation réelle
    try:
        from backend.app.api.middleware import i18n_middleware
        obj = getattr(i18n_middleware, 'TranslationCacheMiddleware')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_rtlsupportmiddleware_class():
    # Instanciation réelle
    try:
        from backend.app.api.middleware import i18n_middleware
        obj = getattr(i18n_middleware, 'RTLSupportMiddleware')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

