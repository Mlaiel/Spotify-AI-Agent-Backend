# Spotify AI Agent – Database Scripts

## Overview
This directory contains all production-ready database scripts for initialization, migration, backup, restore, and seeding. Optimized for security, compliance, ML/AI, auditing, and business logic.

### Features & Best Practices
- **backup_db.sh**: full backup for Postgres, MongoDB, Redis
- **restore_db.sh**: restore all databases from backup
- **init_db.py**: initializes DBs, users, security policies
- **migrate_db.py**: runs migrations (Alembic, Mongo, Elastic)
- **seed_data.py**: seed data for dev, test, demo
- Logging, error handling, security, compliance
- Extensible for more DBs, audits, monitoring

### Recommendations
- Automate and test backups regularly
- Version and review all migrations
- Maintain seed data for ML/AI tests and demo
- Integrate security and compliance checks in CI/CD

### Authors & Roles
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect

---
**See the individual scripts and the project checklist for details.**
