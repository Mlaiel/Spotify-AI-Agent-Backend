from unittest.mock import Mock

import logging
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def alembic_logger():
    logger = logging.getLogger("alembic.test")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][ALEMBIC-TEST] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

@pytest.fixture(scope="session")
def alembic_engine():
    """Initialise une base temporaire pour les tests Alembic."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()

@pytest.fixture(scope="function")
def alembic_session(alembic_engine):
    """Session SQLAlchemy isolée pour chaque test Alembic."""
    Session = sessionmaker(bind=alembic_engine)
    session = Session()
    yield session
    session.close()
