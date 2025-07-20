from unittest.mock import Mock

import logging
import pytest
from unittest.mock import MagicMock

def mongo_root_logger():
    logger = logging.getLogger("mongo.root.test")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][MONGO-ROOT-TEST] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

@pytest.fixture(scope="session")
def mongo_root_client():
    """Mock MongoDB global pour tous les tests racine."""
    client = MagicMock()
    client.db = MagicMock()
    client.db.command = MagicMock()
    client.db.create_collection = MagicMock()
    client.db.list_collection_names = MagicMock()
    return client
