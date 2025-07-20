"""
Alembic migration environment for Spotify AI Agent

This file configures the Alembic context for migrations, including connection to the PostgreSQL database, model metadata import, migration logic, advanced logging, audit, and security.

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect
"""

import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import logging

# Add app to sys.path for model imports
dirs = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')),
]
for d in dirs:
    if d not in sys.path:
        sys.path.append(d)

# Dynamically discover all model metadata
from sqlalchemy.ext.declarative import DeclarativeMeta
import importlib
import pkgutil

def discover_metadata():
    meta = []
    try:
        models_pkg = importlib.import_module('app.models')
        for _, modname, _ in pkgutil.iter_modules(models_pkg.__path__):
            mod = importlib.import_module(f'app.models.{modname}')
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, DeclarativeMeta):
                    if hasattr(obj, 'metadata'):
                        meta.append(obj.metadata)
    except Exception as e:
        logging.warning(f"Metadata discovery failed: {e}")
    return meta

target_metadata = discover_metadata()

# Alembic Config object
config = context.config
fileConfig(config.config_file_name)
logger = logging.getLogger('alembic.env')

# Security: Mask DB URL in logs
def mask_url(url):
    if '@' in url:
        return url.split('@')[0].split('//')[0] + '://***:***@' + url.split('@')[1].split(':')[0] + ':***'
    return url

# Audit: Log migration events
def log_event(event):
    logger.info(f"[MIGRATION AUDIT] {event}")

# Partitioning/Trigger support: Add hooks here if needed

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    log_event(f"Running offline migration on {mask_url(url)}")
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True, compare_type=True)
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config()
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        log_event(f"Running online migration on {mask_url(str(connection.engine.url)}")
        context.configure(
            connection=connection, target_metadata=target_metadata, compare_type=True)
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
