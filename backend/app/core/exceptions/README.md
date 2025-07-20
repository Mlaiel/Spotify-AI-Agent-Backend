# Exceptions Module – Spotify AI Agent (EN)

This module centralizes all business, API, AI, security, database, and Spotify exceptions for an industrial, secure, and observable backend.

## Creator Team (roles)
✅ Lead Dev + AI Architect  
✅ Senior Backend Developer (Python/FastAPI/Django)  
✅ Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)  
✅ DBA & Data Engineer (PostgreSQL/Redis/MongoDB)  
✅ Backend Security Specialist  
✅ Microservices Architect  

## Submodules
- **base_exceptions.py**: Hierarchy, logging, code, i18n, audit
- **api_exceptions.py**: HTTP, validation, throttling, payload, FastAPI/Django
- **auth_exceptions.py**: Auth, permissions, JWT, OAuth, MFA, security
- **database_exceptions.py**: SQL, NoSQL, transaction, integrity, timeouts, audit
- **ai_exceptions.py**: Models, prompts, pipeline, quota, explainability, monitoring
- **spotify_exceptions.py**: Spotify API, quotas, rights, integration, business

## Security & Compliance
- All exceptions are logged, auditable, i18n-ready
- No sensitive messages hardcoded, standardized error codes

## Example usage
```python
from core.exceptions import APIException, DatabaseException, AIException
raise APIException("Custom API error", code=418)
```

## See also
- README.fr.md (FR)
- README.de.md (DE)

