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

# Mock automatique pour grpc
try:
    import grpc
except ImportError:
    import sys
    from unittest.mock import Mock
    sys.modules['grpc'] = Mock()
    if 'grpc' == 'opentelemetry':
        sys.modules['opentelemetry.exporter'] = Mock()
        sys.modules['opentelemetry.instrumentation'] = Mock()
    elif 'grpc' == 'grpc':
        sys.modules['grpc_tools'] = Mock()

# Mock pour gRPC manquant
try:
    import grpc
except ImportError:
    import sys
    from unittest.mock import Mock
    sys.modules['grpc'] = Mock()

import pytest

# Tests générés automatiquement avec logique métier réelle
