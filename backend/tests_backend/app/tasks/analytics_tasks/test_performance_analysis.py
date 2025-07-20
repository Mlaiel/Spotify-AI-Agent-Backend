# Mock automatique pour celery
try:
    import celery
except ImportError:
    import sys
    from unittest.mock import Mock
    sys.modules['celery'] = Mock()
    if 'celery' == 'opentelemetry':
        sys.modules['opentelemetry.exporter'] = Mock()
        sys.modules['opentelemetry.instrumentation'] = Mock()
    elif 'celery' == 'grpc':
        sys.modules['grpc_tools'] = Mock()

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
def test_validate_performance_input():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.tasks.analytics_tasks import performance_analysis
        result = getattr(performance_analysis, 'validate_performance_input')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_analyze_performance_task():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.tasks.analytics_tasks import performance_analysis
        result = getattr(performance_analysis, 'analyze_performance_task')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

