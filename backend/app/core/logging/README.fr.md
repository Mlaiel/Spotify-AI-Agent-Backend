# Spotify AI Agent – Module Logging (FR)

Ce module fournit un système de logging et de monitoring industriel, clé en main, pour plateformes IA, SaaS et microservices.

## Fonctionnalités
- Configuration centralisée et dynamique (JSON, rotation, Sentry-ready)
- Logging structuré (JSON, context, correlation/trace ID)
- Logging de performance (latence, throughput, Prometheus-ready)
- Audit logging (RGPD/SOX, sécurité, actions IA)
- Suivi d’erreurs (Sentry/ELK, enrichissement contextuel)
- Agrégation de logs (multi-service, export JSON/CSV)
- Logging asynchrone (FastAPI, Celery, streaming)

## Fichiers clés
- `logger_config.py` : Config centrale, JSON/rotation/Sentry
- `structured_logger.py` : Logging structuré/context
- `performance_logger.py` : Latence/throughput, décorateurs
- `audit_logger.py` : Audit trail, conformité, sécurité
- `error_tracker.py` : Suivi d’erreurs, Sentry/ELK
- `log_aggregator.py` : Agrégation, export, multi-service
- `async_logger.py` : Logging asynchrone microservices
- `__init__.py` : Expose tous les modules pour import direct

## Exemple d’utilisation
```python
from .logger_config import setup_logging
from .structured_logger import StructuredLogger
setup_logging()
logger = StructuredLogger()
logger.info("Connexion utilisateur", context={"user_id": 123})
```

## Prêt pour la production
- Typage strict, gestion d’erreur robuste
- Aucun TODO, aucune logique manquante
- Intégrable dans APIs, microservices, pipelines analytics
- Extensible Sentry, ELK, Prometheus, Datadog

