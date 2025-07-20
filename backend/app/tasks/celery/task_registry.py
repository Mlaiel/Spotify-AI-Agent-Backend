"""
Task Registry
-------------
Dynamische, auditierbare Registrierung aller Celery-Tasks im System.
- Auto-Discovery, Versionierung, Security, Logging
"""
import importlib
import pkgutil
import logging
from celery import Celery

def register_all_tasks(celery_app: Celery, base_package: str = "app.tasks"):
    """Registriert alle Tasks dynamisch aus allen Submodulen (auto-discovery)."""
    for finder, name, ispkg in pkgutil.walk_packages(["/workspaces/Achiri/spotify-ai-agent/backend/app/tasks"], base_package + "."):
        try:
            importlib.import_module(name)
            logging.info(f"Registered tasks from {name}")
        except Exception as e:
            logging.error(f"Failed to register tasks from {name}: {e}")
    # ... Audit-Log, Versionierung, Security ...
