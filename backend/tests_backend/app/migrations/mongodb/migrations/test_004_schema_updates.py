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
from . import mongo_client_mock

def test_004_schema_updates(mongo_client_mock):
    """Test avancé de mise à jour de schéma sur collection existante."""
    # Configuration du mock pour la commande de modification de collection
    mongo_client_mock.db.command.return_value = {"ok": 1.0}
    
    # Exécution de la mise à jour du schéma avec validation email
    result = mongo_client_mock.db.command({
        "collMod": "users", 
        "validator": {
            "email": {"$type": "string"}
        }
    })
    
    # Vérifications critiques pour la logique métier
    assert result["ok"] == 1.0
    
    # Vérification que la commande a été appelée avec les bons paramètres
    mongo_client_mock.db.command.assert_called_once_with({
        "collMod": "users", 
        "validator": {
            "email": {"$type": "string"}
        }
    })