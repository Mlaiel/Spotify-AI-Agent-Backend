# MongoDB Migration Documentation (EN)

**Created by: Spotify AI Agent Core Team**
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

---

## Purpose
This module contains all scripts and versioned migrations for MongoDB collections, indexes, advanced data, compliance, and automation for the Spotify AI Agent backend. All scripts are production-ready, business-aligned, and CI/CD-integrated.

## Structure
- `migrations/` : All migration scripts (collections, indexes, data, schema updates, advanced, rollback, partitioning, bulk, health-check, GDPR, zero-downtime)
- `migrate.py` : Automation script to run all migrations in order (core & advanced)
- `__init__.py` : Auto-discovery, governance, security, and helper functions for automation

## Advanced Features & Best Practices
- All scripts are idempotent, versioned, and CI/CD-ready
- Security: Use environment variables for credentials and endpoints, audit logging, rollback, compliance checks
- Compliance: GDPR, SOC2, ISO 27001, audit trails, consent management, automated erasure
- Automation: Auto-discovery, dry-run, audit, health-check, bulk import/export, partitioning, zero-downtime
- Multilingual, geo, AI/ML, analytics, versioning, and business logic support
- All changes are tested in staging before production

## Usage Example
### Run All Migrations (Core & Advanced)
```bash
python migrate.py --env=production
```

### Run a single migration manually
```bash
mongo < migrations/005_create_advanced_collections.js
```

### Bulk Import/Export
```bash
mongo < migrations/bulk_import_export.js --eval 'var mode="import"; var collection="users"; var file="users.json"'
```

### Health-Check
```bash
mongo < migrations/health_check.js
```

### GDPR Erasure
```bash
mongo < migrations/gdpr_erasure.js --eval 'var userId="..."'
```

### Zero-Downtime Migration
```bash
mongo < migrations/zero_downtime_migration.js
```

### Rollback
```bash
mongo < migrations/rollback.js --eval 'var targetVersion="004"'
```

## Query Examples
- Find all audit logs for a user:
  ```js
  db.audit_log.find({entity_type: "user", entity_id: ObjectId("...")})
  ```
- List all users who have not granted consent:
  ```js
  db.consent.find({granted: false})
  ```
- Geo search for events:
  ```js
  db.geo_events.find({location: {$near: {$geometry: {type: "Point", coordinates: [lng, lat]}, $maxDistance: 10000}}})
  ```
- Multilingual content by language:
  ```js
  db.multilingual_content.find({lang: "en"})
  ```

## Governance & Extension
- All scripts must be reviewed and approved by the Core Team
- New scripts must follow naming/versioning conventions and include docstrings
- Security and compliance checks are mandatory for all migrations

## Contact
For changes, incidents, or questions, contact the Core Team via Slack #spotify-ai-agent or GitHub. For security/compliance, escalate to the Security Officer.

---

*This documentation is auto-generated and maintained as part of the CI/CD pipeline. Last update: July 2025.*

