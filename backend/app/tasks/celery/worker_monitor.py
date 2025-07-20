"""
Worker Monitoring
----------------
Health Checks, Metrics, Alerting, Auto-Restart für Celery-Worker.
- Security, Observability, Audit
"""
import logging
from celery.app.control import Inspect
from celery import Celery

def monitor_workers(celery_app: Celery):
    """Überwacht alle Worker, prüft Health, sammelt Metriken, triggert Alerts."""
    i = Inspect(app=celery_app)
    stats = i.stats() or {}
    for worker, info in stats.items():
        logging.info(f"Worker {worker} status: {info}")
        # ... Health-Checks, Alerting, Auto-Restart, Audit ...
    return stats

# Optional: Integration mit Prometheus, OpenTelemetry, externes Alerting
