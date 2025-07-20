from unittest.mock import Mock

import pytest
from unittest.mock import MagicMock
import logging

def mongo_migration_logger():
    logger = logging.getLogger("mongo.migration.test")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][MONGO-MIGRATION-TEST] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

@pytest.fixture(scope="session")
def mongo_client_mock():
    """Mock MongoDB pour tests de migrations."""
    client = MagicMock()
    client.db = MagicMock()
    client.db.command = MagicMock()
    client.db.create_collection = MagicMock()
    client.db.list_collection_names = MagicMock()
    return client
