import logging
from typing import Callable, List

logger = logging.getLogger("invalidation_service")

class InvalidationService:
    """
    Service d’invalidation de cache avancé : granularité, pub/sub, hooks, triggers Celery, support multi-backend.
    Permet la synchronisation temps réel entre microservices, IA, analytics, etc.
    """
    def __init__(self, pubsub=None):
        self.hooks: List[Callable[str], None] = []
        self.pubsub = pubsub  # Peut être un client Redis ou autre
    def register_hook(self, hook: Callable[[str], None]):
        self.hooks.append(hook)
        logging.info(f"Hook d’invalidation enregistré: {hook}")
    def invalidate(self, key: str):
        logging.info(f"Invalidation du cache pour la clé: {key}")
        for hook in self.hooks:
            hook(key)
        # Pub/Sub Redis (si configuré)
        if self.pubsub:
            self.pubsub.publish("cache:invalidate", key)
            logging.info(f"Message pubsub envoyé pour {key}")
        # Trigger Celery (exemple)
        # from tasks.invalidate import invalidate_cache_task
        # invalidate_cache_task.delay(key)
