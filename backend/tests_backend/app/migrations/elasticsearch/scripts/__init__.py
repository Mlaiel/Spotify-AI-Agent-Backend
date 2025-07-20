from unittest.mock import Mock

import pytest
from unittest.mock import MagicMock
import logging

def es_script_logger():
    logger = logging.getLogger("es.script.test")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][ES-SCRIPT-TEST] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

@pytest.fixture(scope="session")
def es_script_mock():
    """Mock Elasticsearch pour tests de scripts."""
    es = MagicMock()
    es.indices = MagicMock()
    es.reindex = MagicMock()
    es.search = MagicMock()
    es.index = MagicMock()
    return es
