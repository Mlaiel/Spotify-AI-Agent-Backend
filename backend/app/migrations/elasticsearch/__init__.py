"""
Elasticsearch migration package for Spotify AI Agent

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

This package contains all mappings, scripts, and automation for managing Elasticsearch indexes, data, and migrations. Advanced features: multilingual, geo, nested, audit, compliance, lifecycle, health-check, bulk, rollback, governance.

Governance: All changes are peer-reviewed, versioned, and require business/security justification. Multilingual, geo, nested, audit, compliance, lifecycle, health-check und rollback sind Standard.
"""

import os

def discover_submodules():
    base = os.path.dirname(__file__)
    return [f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f)) and not f.startswith('__')]

# Example usage:
# for submodule in discover_submodules():
#     print(submodule)
