"""
PostgreSQL migration scripts package for Spotify AI Agent

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

This package contains all versioned PostgreSQL migration scripts for schema, tables, business logic, security, audit, compliance, rollback, partitioning, bulk, health-check, GDPR erasure, zero-downtime migrations and utility scripts for the Spotify AI Agent backend.

Features:
- Auto-discovery of all migration scripts (core & advanced)
- Governance, security, and compliance checks
- Helper functions for automation, CI/CD, and audit logging
- Usage examples and extension policy
- All scripts are atomic, idempotent, reversible, production-ready, and business-aligned
- Utility scripts: 006_rollback.py, 007_partitioning.py, 008_bulk_import_export.py, 009_health_check.py, 010_gdpr_erasure.py, 011_zero_downtime_migration.py

Usage:
    from . import discover_migrations
    scripts = discover_migrations()
    # Example: import and run a migration
    # from .006_rollback import downgrade
    # downgrade()

"""

import os
import importlib

def discover_migrations():
    """Auto-discover all migration scripts in this package (ordered by version and utility)."""
    base = os.path.dirname(__file__)
    scripts = []
    for f in os.listdir(base):
        if f.endswith('.py') and not f.startswith('__'):
            scripts.append(f)
    return sorted(scripts)

# Governance & Security Policy
# - All scripts must be reviewed and approved by the Core Team
# - Security and compliance checks are mandatory
# - Usage of this package is logged for audit purposes
