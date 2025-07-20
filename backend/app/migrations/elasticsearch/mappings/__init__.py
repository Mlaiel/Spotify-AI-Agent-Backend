"""
Elasticsearch mappings package for Spotify AI Agent

Created by: Spotify AI Agent Core Team
- Lead Dev + Architecte IA
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

This package contains all Elasticsearch index mappings for advanced search, analytics, AI, security, audit, compliance, lifecycle management, multilingual content, synonyms, geo, geo-shape, and nested objects.

Governance: All mappings are peer-reviewed, versioned, and require business/security justification for changes. Audit, consent, GDPR/DSGVO, versioning, lifecycle, multilingual, synonym, geo, geo-shape und nested sind Standard.
"""

import os
import json

def discover_mappings():
    base = os.path.dirname(__file__)
    mappings = []
    for f in os.listdir(base):
        if f.endswith('_mapping.json'):
            mappings.append(f)
    return sorted(mappings)

# Example usage:
# for mapping_file in discover_mappings():
#     with open(os.path.join(os.path.dirname(__file__), mapping_file) as f:
#         mapping = json.load(f)
