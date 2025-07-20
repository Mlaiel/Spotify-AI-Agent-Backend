from unittest.mock import Mock

import pytest
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def pg_logger():
    logger = logging.getLogger("pg.migration.test")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][PG-MIGRATION-TEST] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

@pytest.fixture(scope="session")
def pg_engine():
    """Initialise une base temporaire PostgreSQL pour les tests de migration."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()

@pytest.fixture(scope="function")
def pg_session(pg_engine):
    """Session SQLAlchemy isolée pour chaque test."""
    Session = sessionmaker(bind=pg_engine)
    session = Session()
    yield session
    session.close()
