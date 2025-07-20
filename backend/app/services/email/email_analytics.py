import logging
from typing import List, Dict, Optional

logger = logging.getLogger("email_analytics")

class EmailAnalytics:
    """
    Service d’analytics email avancé : tracking open/click, scoring ML, logs, hooks, sécurité, observabilité.
    Utilisé pour campagnes, notifications, IA, scoring, reporting, etc.
    """
    def __init__(self):
        self.events: List[Dict] = []  # Stocke les événements (open, click, etc.)
        self.hooks = []
    def register_hook(self, hook):
        self.hooks.append(hook)
        logger.info(f"Analytics hook enregistré: {hook}")
    def track_send(self, to: List[str], subject: str, metadata: Optional[Dict] = None):
        event = {"type": "send", "to": to, "subject": subject, "metadata": metadata}
        self.events.append(event)
        logger.info(f"[ANALYTICS] Email envoyé à {to} | Sujet: {subject}")
        for hook in self.hooks:
            hook(event)
    def track_open(self, email_id: str, user_id: str):
        event = {"type": "open", "email_id": email_id, "user_id": user_id}
        self.events.append(event)
        logger.info(f"[ANALYTICS] Email ouvert: {email_id} par {user_id}")
        for hook in self.hooks:
            hook(event)
    def track_click(self, email_id: str, user_id: str, link: str):
        event = {"type": "click", "email_id": email_id, "user_id": user_id, "link": link}
        self.events.append(event)
        logger.info(f"[ANALYTICS] Lien cliqué: {link} dans {email_id} par {user_id}")
        for hook in self.hooks:
            hook(event)
    def ml_score(self, email_id: str) -> float:
        # Exemple: scoring ML fictif (à remplacer par un vrai modèle)
        score = 0.95
        logger.info(f"[ML] Score de délivrabilité pour {email_id}: {score}")
        return score
