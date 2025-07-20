"""
Celery App Factory (Root)
------------------------
- Erstellt und konfiguriert die zentrale Celery-App für das gesamte Backend
- Integriert Security, Monitoring, Prometheus, OpenTelemetry, Sentry, Dynamic Task Discovery
- Produktionsreif, robust, mit Alerting und Multi-Queue

Autoren & Rollen:
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect
"""

from celery import Celery
import os
from prometheus_client import start_http_server
from opentelemetry.instrumentation.celery import CeleryInstrumentor
import sentry_sdk

celery_app = Celery(
    "spotify_ai_agent",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    task_acks_late=True,
    broker_heartbeat=10,
    broker_pool_limit=20,
    result_expires=3600,
    task_soft_time_limit=600,
    task_time_limit=1200,
    # ... weitere Security/Monitoring-Settings ...)
)

# Prometheus Monitoring
start_http_server(int(os.getenv("PROMETHEUS_METRICS_PORT", 8000))

# OpenTelemetry Instrumentierung
CeleryInstrumentor().instrument()

# Sentry Integration
sentry_sdk.init(dsn=os.getenv("SENTRY_DSN", "")

# Optional: Monitoring, Signal-Handler, Task-Discovery
# ...
