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
def test_003_add_ai_tables(pg_engine):
    """Test clé en main d’ajout de tables AI (ai_model, ai_training_log)."""
    inspector = inspect(pg_engine)
    for table in ["ai_model", "ai_training_log"]:
        assert table in inspector.get_table_names()
