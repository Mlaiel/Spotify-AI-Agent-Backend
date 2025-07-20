# Spotify AI Agent – Core Configuration (EN)

## Overview
This directory contains all configuration modules for the Spotify AI Agent backend. Each config is production-ready, secure, validated, and business-exploitable. No TODOs or placeholders.

---

## Architecture
- **settings.py**: Central Pydantic config, loads from .env, validates all critical settings
- **ai_config.py**: AI/ML models, providers, moderation, security
- **database_config.py**: PostgreSQL, MongoDB, Redis, pooling, security
- **environment_config.py**: Environment, debug, version, region, timezone
- **redis_config.py**: Redis advanced config (cluster, SSL, timeouts)
- **security_config.py**: JWT, CORS, CSP, brute-force, security policies
- **spotify_config.py**: Spotify API integration (OAuth2, scopes, endpoints)

---

## Security & Compliance
- All secrets loaded from environment variables or .env
- Pydantic validation for all configs
- No sensitive data hardcoded

## Extensibility
- Each config is modular and can be extended per environment
- Ready for CI/CD, cloud, and microservices

## Example Usage
```python
from core.config import settings, AIConfig, DatabaseConfig, SecurityConfig
print(settings.secret_key)
```

---

## See Also
- [README.fr.md](./README.fr.md) (Français)
- [README.de.md](./README.de.md) (Deutsch)

