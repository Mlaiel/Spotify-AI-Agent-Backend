# Spotify AI Agent – Requirements

## Overview
This directory contains all dependency lists for development, production, security, testing, and compliance. Separation by environment and role ensures maximum security, maintainability, scalability, and business compliance.

### Structure & Best Practices
- **base.txt**: core dependencies for backend, ML/AI, security, DB, API, microservices
- **development.txt**: dev tools, linters, debuggers, test, mocking
- **production.txt**: only stable, production-ready packages
- **security.txt**: security, audit, compliance, secrets management, vulnerability scan
- **testing.txt**: test, mocking, coverage, integration, load, ML/AI tests

#### Recommendations
- Regularly check dependencies with `safety` and `pip-audit`
- Strictly separate dev/prod (no debug/dev tools in production!)
- Modularize ML/AI serving and data engineering packages as needed
- Automate security and compliance scans (CI/CD)
- Version and review all changes (code review, audit)

### Authors & Roles
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect

---
**See the individual requirements files and the project checklist for details.**
