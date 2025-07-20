"""
Spotify AI Agent ORM Root Package

Lead Dev: [Name], Architecte IA: [Name], Backend: [Name], ML: [Name], DBA: [Name], Security: [Name], Microservices: [Name]

Dieses Paket enthält alle Basisklassen, Mixins und Governance für produktionsreife, auditierbare, DSGVO-konforme ORM-Modelle:
- Basisklassen für Validierung, Security, Audit, Soft-Delete, Timestamps, Multi-Tenancy, Data Lineage
- Mixins für Versionierung, Traceability, Compliance, Logging, User-Attribution, Explainability
- Governance, Extension Policy, Security, Compliance, CI/CD, Data Lineage

Alle Submodule (ai, analytics, collaboration, spotify, users) sind für PostgreSQL, MongoDB und hybride Architekturen optimiert.

"""

from .base_model import *
from .mixins import *