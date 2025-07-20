"""
README – Alembic Migration Scripts (EN)

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

This directory contains all versioned Alembic migration scripts for the PostgreSQL database schema. Each file represents a single, atomic, and reversible schema change. Scripts are named with a timestamp and a clear description.

## Best Practices
- Each migration is peer-reviewed, CI/CD-tracked, and tested in staging before production.
- Use the provided template for new migrations (see `template.py`).
- All migrations are auditable and include rollback logic.
- Security, audit, analytics, and compliance tables are included for enterprise readiness.
- Indexes, constraints, and partitioning are recommended for performance and compliance.

## Usage Example
```bash
alembic upgrade head
alembic downgrade -1
```

## Security & Governance
- All migrations are versioned and peer-reviewed
- Security and compliance migrations are included for audit and GDPR/DSGVO
- Migration changes require business and security review

For questions or changes, contact the Core Team via Slack #spotify-ai-agent or GitHub.
"""
