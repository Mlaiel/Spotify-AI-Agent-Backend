import logging
from typing import List, Dict, Optional

logger = logging.getLogger("notification_service")

class NotificationService:
    """
    Service de notifications avancé pour la collaboration : in-app, email, push, hooks, audit, sécurité, observabilité.
    Utilisé pour inviter, notifier, alerter dans les workflows IA, Spotify, analytics, etc.
    """
    def __init__(self):
        self.hooks = []  # Hooks pour extension (ex: analytics, audit, IA)
    def register_hook(self, hook):
        self.hooks.append(hook)
        logger.info(f"Notification hook enregistré: {hook}")
    def send(self, user_id: str, message: str, channels: Optional[List[str]] = None, metadata: Optional[Dict] = None):
        channels = channels or ["in-app"]
        for channel in channels:
            logger.info(f"Notification envoyée à {user_id} via {channel}: {message}")
            # Ici, intégrer email, push, websocket, etc.
        for hook in self.hooks:
            hook(user_id, message, channels, metadata)
    def audit(self, user_id: str, message: str):
        logger.info(f"[AUDIT] Notification à {user_id}: {message}")
    def notify_collaboration_invite(self, user_id: str, workspace_id: str):
        msg = f"Vous avez été invité à collaborer sur le workspace {workspace_id}"
        self.send(user_id, msg)
        self.audit(user_id, msg)
