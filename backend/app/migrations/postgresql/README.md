# Documentation (EN)

# PostgreSQL Migration Documentation (EN)

**Created by: Spotify AI Agent Core Team**
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

---

## Purpose
This module contains all scripts and versioned migrations for PostgreSQL schema, tables, advanced data, compliance, and automation for the Spotify AI Agent backend. All scripts are production-ready, business-aligned, and CI/CD-integrated.

## Structure
- `versions/` : All migration scripts (schema, tables, business logic, advanced, rollback, partitioning, bulk, health-check, GDPR, zero-downtime)
- `env.py`, `alembic.ini`, `script.py.mako` : Alembic configuration and templates
- `__init__.py` : Auto-discovery, governance, security, and helper functions for automation

## Advanced Features & Best Practices
- All scripts are atomic, idempotent, reversible, and CI/CD-ready
- Security: audit, logs, access control, GDPR anonymization, rollback, compliance checks
- Compliance: GDPR, SOC2, ISO 27001, audit trails, consent management, automated erasure
- Automation: auto-discovery, dry-run, audit, health-check, bulk import/export, partitioning, zero-downtime
- Multilingual, geo, AI/ML, analytics, versioning, and business logic support
- All changes are tested in staging before production

## Usage Example
### Run All Migrations (Core & Advanced)
```bash
alembic upgrade head
```

### Run a single migration manually
```bash
python versions/007_partitioning.py
```

### Bulk Import/Export
```python
from versions.008_bulk_import_export import bulk_import, bulk_export
bulk_import('analytics', 'analytics.csv')
bulk_export('users', 'users.csv')
```

### Health-Check
```python
from versions.009_health_check import check_health
check_health()
```

### GDPR Erasure
```python
from versions.010_gdpr_erasure import erase_user
erase_user(user_id=123)
```

### Zero-Downtime Migration
```python
from versions.011_zero_downtime_migration import upgrade
upgrade()
```

### Rollback
```bash
python versions/006_rollback.py
```

## Query Examples
- Find all audit logs for a user:
  ```sql
  SELECT * FROM audit_log WHERE entity_type = 'user' AND entity_id = 123;
  ```
- List all users who have not granted consent:
  ```sql
  SELECT * FROM consent WHERE granted = false;
  ```
- Analytics by month:
  ```sql
  SELECT date_trunc('month', created_at) AS month, COUNT(*) FROM analytics GROUP BY month;
  ```

## Governance & Extension
- All scripts must be reviewed and approved by the Core Team
- New scripts must follow naming/versioning conventions and include docstrings
- Security and compliance checks are mandatory for all migrations

## Contact
For changes, incidents, or questions, contact the Core Team via Slack #spotify-ai-agent or GitHub. For security/compliance, escalate to the Security Officer.

---

*This documentation is auto-generated and maintained as part of the CI/CD pipeline. Last update: July 2025.*

