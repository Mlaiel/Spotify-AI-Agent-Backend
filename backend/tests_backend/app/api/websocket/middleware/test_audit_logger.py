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

# Mock automatique pour opentelemetry
try:
    import opentelemetry
except ImportError:
    import sys
    from unittest.mock import Mock
    sys.modules['opentelemetry'] = Mock()
    if 'opentelemetry' == 'opentelemetry':
        sys.modules['opentelemetry.exporter'] = Mock()
        sys.modules['opentelemetry.instrumentation'] = Mock()
    elif 'opentelemetry' == 'grpc':
        sys.modules['grpc_tools'] = Mock()

# Mock pour OpenTelemetry manquant
try:
    import opentelemetry.exporter
except ImportError:
    import sys
    from unittest.mock import Mock
    sys.modules['opentelemetry'] = Mock()
    sys.modules['opentelemetry.exporter'] = Mock()

import pytest

# Tests générés automatiquement avec logique métier réelle
def test_auditlogger_class():
    # Instanciation réelle
    try:
        from backend.app.api.websocket.middleware import audit_logger
        obj = getattr(audit_logger, 'AuditLogger')()
        assert obj is not None
    except Exception as exc:
        pytest.fail('Erreur lors de l\'instanciation réelle : {}'.format(exc))

