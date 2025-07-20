"""
Celery App Factory
-----------------
Erstellt und konfiguriert die zentrale Celery-App für das gesamte Backend.
- Security, Monitoring, Dynamic Task Discovery, Multi-Queue
"""
from celery import Celery
import os

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
    # Additional Security/Monitoring settings
)

# Optional: Monitoring, Signal-Handler, Task-Discovery
# ...
