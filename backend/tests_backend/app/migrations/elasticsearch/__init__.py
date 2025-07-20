from unittest.mock import Mock

import logging
import pytest
from unittest.mock import MagicMock

def es_root_logger():
    logger = logging.getLogger("es.root.test")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][ES-ROOT-TEST] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

@pytest.fixture(scope="session")
def es_root_mock():
    """Mock Elasticsearch global pour tous les tests racine."""
    es = MagicMock()
    es.indices = MagicMock()
    es.reindex = MagicMock()
    es.search = MagicMock()
    es.index = MagicMock()
    return es
