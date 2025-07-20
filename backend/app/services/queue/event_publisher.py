import logging
from typing import Callable, Dict, Any, List

logger = logging.getLogger("event_publisher")

class EventPublisher:
    """
    Service de publication d’événements avancé : pub/sub, hooks, audit, sécurité, observabilité.
    Utilisé pour diffuser des événements IA, analytics, Spotify, collaboration, etc.
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[Any], None]] = {}
        self.hooks: List[Callable] = []
    def subscribe(self, event_type: str, callback: Callable[[Any], None]):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        logger.info(f"Abonné à {event_type}: {callback}")
    def publish(self, event_type: str, data: Any):
        logger.info(f"Publication événement {event_type}: {data}")
        for callback in self.subscribers.get(event_type, []):
            callback(data)
        for hook in self.hooks:
            hook(event_type, data)
        self.audit(event_type, data)
    def register_hook(self, hook: Callable):
        self.hooks.append(hook)
        logger.info(f"EventPublisher hook enregistré: {hook}")
    def audit(self, event_type: str, data: Any):
        logger.info(f"[AUDIT] Event {event_type} publié: {data}")
