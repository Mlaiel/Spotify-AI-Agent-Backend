# Spotify AI Agent – Utils Modul (DE)

Dieses Modul bietet ein industrietaugliches Utility-Toolkit für KI-, SaaS- und Microservices-Plattformen.

## Features
- Async-Utilities (Retry, Timeout, Executor)
- Crypto-Utilities (Hash, HMAC, Random, Signatur)
- Date-Utilities (Parsing, Formatierung, Humanize, Zeitzone)
- Decorators (Exception, Timing, Retry, Logging)
- Env-Utilities (dotenv, Validierung, Fallback, Secrets)
- File-Utilities (Upload, Validierung, S3, Temp, Sicherheit)
- Helpers (Flatten, Chunk, Deep_Get, Safe_Cast)
- ID-Utilities (UUID, ShortID, NanoID)
- Serializers (JSON, Dict, Model, FastAPI/Pydantic-ready)
- String-Utilities (Slugify, Random, Clean, Truncate)
- Validatoren (Email, URL, Phone, IBAN, Custom)
- `__init__.py`: Stellt alle Module für den Direktimport bereit

## Beispiel
```python
from .date_utils import format_date
from .validators import is_email
format_date(now_utc(), locale="de", tz="Europe/Berlin")
is_email("foo@bar.com")
```

## Produktionsbereit
- 100% typisiert, robuste Fehlerbehandlung
- Keine TODOs, keine Platzhalter
- In APIs, Microservices, Analytics-Pipelines integrierbar
- Erweiterbar für neue Business-Anforderungen

