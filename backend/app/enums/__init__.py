"""
Spotify AI Agent Enums Package

Created by: Spotify AI Agent Core Team
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect

This package centralizes all enums for strict typing, validation, business logic, security, and compliance across the backend.

Governance: All enums are peer-reviewed, versioned, and require business/security justification for changes.
"""

import os
import importlib

def discover_enums():
    base = os.path.dirname(__file__)
    enums = []
    for f in os.listdir(base):
        if f.endswith('_enums.py'):
            enums.append(f[:-3])
    return sorted(enums)

# Example usage:
# for enum_mod in discover_enums():
#     mod = importlib.import_module(f".enums.{enum_mod}", package=__package__)