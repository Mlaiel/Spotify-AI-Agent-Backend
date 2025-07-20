"""
Celery Configuration
-------------------
Zentrale, sichere, umgebungsgetriebene Konfiguration für Celery.
- Security, Monitoring, Production-Ready
"""
import os

class CeleryConfig:
    broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    task_serializer = "json"
    accept_content = ["json"]
    result_serializer = "json"
    timezone = "UTC"
    enable_utc = True
    task_track_started = True
    worker_max_tasks_per_child = 1000
    worker_disable_rate_limits = False
    task_acks_late = True
    broker_heartbeat = 10
    broker_pool_limit = 20
    result_expires = 3600
    task_soft_time_limit = 600
    task_time_limit = 1200
    # ... weitere Security/Monitoring-Settings ...

# Optional: Load into celery_app.conf.from_object(CeleryConfig)
