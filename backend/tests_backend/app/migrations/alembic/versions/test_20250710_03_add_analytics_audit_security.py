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
from sqlalchemy import inspect

def test_add_analytics_audit_security(db_engine):
    """Vérifie l’ajout des tables analytics, audit et security, et la conformité des logs."""
    inspector = inspect(db_engine)
    # Vérification des tables critiques
    for table in ['analytics', 'audit', 'security']:
        assert table in inspector.get_table_names(), f"La table '{table}' doit exister."
    # Vérification des colonnes d’audit
    audit_columns = [col['name'] for col in inspector.get_columns('audit')]
    assert 'event_type' in audit_columns and 'timestamp' in audit_columns
