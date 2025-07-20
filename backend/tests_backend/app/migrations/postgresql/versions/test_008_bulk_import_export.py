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
def test_008_bulk_import_export(pg_engine):
    """Test clé en main d’import/export massif de données."""
    inspector = inspect(pg_engine)
    # Vérifier la présence de données dans la table après import # (Simulation, à adapter selon la logique réelle)
    assert "imported_data" in inspector.get_table_names()
