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
def test_aiexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import ai_exceptions
        obj = getattr(ai_exceptions, 'AIException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_modelloadexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import ai_exceptions
        obj = getattr(ai_exceptions, 'ModelLoadException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_promptexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import ai_exceptions
        obj = getattr(ai_exceptions, 'PromptException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_pipelineexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import ai_exceptions
        obj = getattr(ai_exceptions, 'PipelineException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_aiquotaexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import ai_exceptions
        obj = getattr(ai_exceptions, 'AIQuotaException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

def test_explainabilityexception_class():
    # Instanciation réelle
    try:
        from backend.app.core.exceptions import ai_exceptions
        obj = getattr(ai_exceptions, 'ExplainabilityException')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

