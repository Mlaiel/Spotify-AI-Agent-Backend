# MongoDB Migration Scripts Documentation (EN)

**Created by: Spotify AI Agent Core Team**
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

---

## Purpose
This directory contains all versioned MongoDB migration scripts for collections, indexes, advanced data transformations, and compliance operations. All scripts are designed for automation, CI/CD, business logic, security, and regulatory compliance (GDPR, SOC2, ISO 27001).

## Scripts Overview
- `001_create_collections.js` – Create all required core collections with validation
- `002_add_indexes.js` – Add indexes for performance, search, analytics
- `003_data_migration.js` – Transform or migrate data for new features
- `004_schema_updates.js` – Update schema validation or add new fields
- `005_create_advanced_collections.js` – Create advanced collections (AI, audit, security, consent, geo, multilingual, versioning)
- `006_add_advanced_indexes.js` – Add advanced indexes (AI, audit, security, consent, geo, multilingual, versioning)
- `rollback.js` – Rollback changes for safe migrations (see below)
- `partitioning.js` – Partition large collections for scalability
- `bulk_import_export.js` – Bulk data import/export for migration and backup
- `health_check.js` – Automated health-checks for collections and indexes
- `gdpr_erasure.js` – GDPR-compliant user data erasure
- `zero_downtime_migration.js` – Zero-downtime migration patterns

> **Note:** All scripts are idempotent, production-ready, and support dry-run and audit modes where applicable.

## Advanced Features & Best Practices
- **Security:**
  - Use environment variables for credentials and endpoints
  - All scripts log actions to `audit_log` and `security_events` collections
  - Built-in validation, error handling, and rollback support
- **Compliance:**
  - GDPR, SOC2, ISO 27001, audit trails, consent management
  - Automated data retention and erasure scripts
- **Automation:**
  - CI/CD-ready, can be run standalone or as part of pipelines
  - Auto-discovery of scripts for migration orchestration
- **Business Logic:**
  - All migrations align with Spotify AI Agent business requirements
  - Support for multilingual, geo, AI/ML, analytics, and versioning
- **Governance:**
  - All changes are tracked, versioned, and auditable
  - Usage policies and extension guidelines included

## Usage Examples
### Run All Migrations (Core & Advanced)
```bash
mongo < 001_create_collections.js
mongo < 002_add_indexes.js
mongo < 003_data_migration.js
mongo < 004_schema_updates.js
mongo < 005_create_advanced_collections.js
mongo < 006_add_advanced_indexes.js
mongo < partitioning.js --eval 'var collection="analytics"; var shardKey="user_id"'
mongo < bulk_import_export.js --eval 'var mode="import"; var collection="users"; var file="users.json"'
mongo < health_check.js
mongo < gdpr_erasure.js --eval 'var userId="..."'
mongo < zero_downtime_migration.js
```

### Rollback Example
```bash
mongo < rollback.js --eval 'var targetVersion="004"'
```

### Bulk Import/Export
```bash
mongoimport --db spotify_ai --collection users --file users.json
mongoexport --db spotify_ai --collection audit_log --out audit_log.json
```

### Health-Check
```bash
mongo < health_check.js
```

### GDPR Erasure
```bash
mongo < gdpr_erasure.js --eval 'var userId="..."'
```

### Zero-Downtime Migration
```bash
mongo < zero_downtime_migration.js
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
  db.multilingual_content.find({lang: "de"})
  ```

## Governance & Extension
- All scripts must be reviewed and approved by the Core Team
- New scripts must follow naming/versioning conventions and include docstrings
- Security and compliance checks are mandatory for all migrations

## Contact
For changes, incidents, or questions, contact the Core Team via Slack #spotify-ai-agent or GitHub. For security/compliance, escalate to the Security Officer.

---

*This documentation is auto-generated and maintained as part of the CI/CD pipeline. Last update: October 2023.*

