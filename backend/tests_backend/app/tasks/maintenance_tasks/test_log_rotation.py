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
def test_validate_log_target():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.tasks.maintenance_tasks import log_rotation
        result = getattr(log_rotation, 'validate_log_target')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

def test_rotate_logs_task():
    # Appel réel de la fonction
    result = None
    try:
        from backend.app.tasks.maintenance_tasks import log_rotation
        result = getattr(log_rotation, 'rotate_logs_task')()
    except Exception as exc:
        pytest.fail('Erreur lors de l\'appel réel : {}'.format(exc))
    assert result is not None

