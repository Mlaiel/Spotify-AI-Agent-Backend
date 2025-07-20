"""
Alembic Migration Environment for Spotify AI Agent

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

This directory contains all Alembic migration scripts for the PostgreSQL database schema.

Governance: All migrations are peer-reviewed, versioned, and require business/security justification for changes. Security, audit, partitioning, compliance und Logging sind Standard.
"""

import os
import importlib

def discover_migrations():
    base = os.path.join(os.path.dirname(__file__), 'versions')
    migrations = []
    for f in os.listdir(base):
        if f.endswith('.py') and not f.startswith('__'):
            migrations.append(f[:-3])
    return sorted(migrations)

# Example usage:
# for migration_mod in discover_migrations():
#     mod = importlib.import_module(f"app.migrations.alembic.versions.{migration_mod}")
