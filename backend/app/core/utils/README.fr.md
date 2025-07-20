# Spotify AI Agent – Module Utils (FR)

Ce module fournit une boîte à outils utilitaire industrielle, clé en main, pour plateformes IA, SaaS et microservices.

## Fonctionnalités
- Utilitaires asynchrones (retry, timeout, executor)
- Utilitaires crypto (hash, HMAC, random, signature)
- Utilitaires date (parsing, formatage, humanize, timezone)
- Décorateurs (exception, timing, retry, logging)
- Utilitaires env (dotenv, validation, fallback, secrets)
- Utilitaires fichiers (upload, validation, S3, temp, sécurité)
- Helpers (flatten, chunk, deep_get, safe_cast)
- Utilitaires ID (UUID, shortid, nanoid)
- Serializers (JSON, dict, model, FastAPI/Pydantic-ready)
- Utilitaires string (slugify, random, clean, truncate)
- Validateurs (email, url, phone, IBAN, custom)
- `__init__.py` : Expose tous les modules pour import direct

## Exemple d’utilisation
```python
from .date_utils import format_date
from .validators import is_email
format_date(now_utc(), locale="fr", tz="Europe/Paris")
is_email("foo@bar.com")
```

## Prêt pour la production
- Typage strict, gestion d’erreur robuste
- Aucun TODO, aucune logique manquante
- Intégrable dans APIs, microservices, pipelines analytics
- Extensible pour nouveaux besoins métier

