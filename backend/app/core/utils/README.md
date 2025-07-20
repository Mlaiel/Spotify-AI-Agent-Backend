# Spotify AI Agent – Utils Module (EN)

This module provides a full-stack, production-grade utility toolkit for AI, SaaS, and microservices platforms.

## Features
- Async utilities (retry, timeout, executor)
- Crypto utilities (hash, HMAC, random, signature)
- Date utilities (parsing, formatting, humanize, timezone)
- Decorators (exception, timing, retry, logging)
- Env utilities (dotenv, validation, fallback, secrets)
- File utilities (upload, validation, S3, temp, security)
- Helpers (flatten, chunk, deep_get, safe_cast)
- ID utilities (UUID, shortid, nanoid)
- Serializers (JSON, dict, model, FastAPI/Pydantic-ready)
- String utilities (slugify, random, clean, truncate)
- Validators (email, url, phone, IBAN, custom)
- `__init__.py`: Exposes all modules for direct import

## Usage Example
```python
from .date_utils import format_date
from .validators import is_email
format_date(now_utc(), locale="fr", tz="Europe/Paris")
is_email("foo@bar.com")
```

## Industrial-Ready
- Strict typing, robust error handling
- No TODOs, no placeholders
- Easily integrable in APIs, microservices, analytics pipelines
- Extensible for new business needs

