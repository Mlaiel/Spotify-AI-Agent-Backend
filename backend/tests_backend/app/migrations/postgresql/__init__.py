from unittest.mock import Mock

import logging
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def pg_root_logger():
    logger = logging.getLogger("pg.root.test")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][PG-ROOT-TEST] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

@pytest.fixture(scope="session")
def pg_root_engine():
    """Initialise une base temporaire PostgreSQL pour tous les tests racine."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()
