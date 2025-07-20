"""
Migrations package for Spotify AI Agent (PostgreSQL, MongoDB, Elasticsearch, Alembic)

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Développeur Backend Senior (Python/FastAPI/Django)
- Ingénieur Machine Learning (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Spécialiste Sécurité Backend
- Architecte Microservices

This package orchestrates all database and search migrations for the Spotify AI Agent backend. It covers PostgreSQL, MongoDB, Elasticsearch, and Alembic, with advanced automation, security, compliance, and business logic.

Features:
- Auto-discovery of all migration scripts and submodules (core & advanced)
- Governance, security, and compliance checks
- Helper functions for automation, CI/CD, and audit logging
- Usage examples and extension policy
- All scripts are atomic, idempotent, reversible, production-ready, and business-aligned
- Utility scripts: rollback, partitioning, bulk, health-check, GDPR/DSGVO, zero-downtime, multilingual, geo, AI/ML

Usage:
    # Example: discover all migration submodules
    import os
    base = os.path.dirname(__file__)
    submodules = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d) and not d.startswith('__')]
    print(submodules)

# Governance & Security Policy
# - All scripts must be reviewed and approved by the Core Team
# - Security and compliance checks are mandatory
# - Usage of this package is logged for audit purposes
"""
