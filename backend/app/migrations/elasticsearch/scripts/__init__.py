"""
Elasticsearch migration scripts package for Spotify AI Agent

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

This package contains all scripts for managing Elasticsearch indexes, mappings, data migrations, multilingual, geo, nested, audit, lifecycle, health-check, and compliance.

Governance: All scripts are peer-reviewed, versioned, and require business/security justification for changes. Multilingual, geo, nested, audit, lifecycle, health-check und compliance sind Standard.
"""

import os

def discover_scripts():
    base = os.path.dirname(__file__)
    scripts = []
    for f in os.listdir(base):
        if f.endswith('.py') and not f.startswith('__'):
            scripts.append(f)
    return sorted(scripts)

# Example usage:
# for script in discover_scripts():
#     print(script)
