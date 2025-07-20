from unittest.mock import Mock

import pytest
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def db_engine():
    """Initialise une base temporaire pour les tests de migration."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()

@pytest.fixture(scope="function")
def db_session(db_engine):
    """Session SQLAlchemy isolée pour chaque test."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()

def log_migration_event(event: str):
    logging.info(f"[MIGRATION-TEST] {event}")
