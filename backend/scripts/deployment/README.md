# Spotify AI Agent – Deployment Scripts

## Overview
This directory contains all production-ready deployment scripts for zero-downtime deployments, health checks, rollbacks, backups, and compliance. Optimized for security, observability, ML/AI, auditing, and business logic.

### Features & Best Practices
- **backup.sh**: full, versioned backup before every deployment
- **deploy.sh**: zero-downtime deployment with health checks, logging, audit
- **health_check.sh**: health check of all core services (backend, celery, redis, postgres, nginx)
- **rollback.sh**: automated, secure rollback to the last backup
- Logging, error handling, security, compliance
- Extensible for blue/green, canary, multi-region

### Recommendations
- Automate and test backups and health checks
- Practice rollbacks regularly (disaster recovery)
- Integrate security and compliance checks in CI/CD
- Keep an audit log for all deployments and rollbacks

### Authors & Roles
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect

---
**See the individual scripts and the project checklist for details.**
