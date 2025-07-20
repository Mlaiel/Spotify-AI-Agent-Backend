import logging
from fastapi import WebSocket
from datetime import datetime
from typing import Optional

logger = logging.getLogger("AuditLogger")

class AuditLogger:
    """
    Logger d'audit pour toutes les connexions, déconnexions et actions sensibles WebSocket.
    Peut être branché sur une base PostgreSQL, MongoDB ou un SIEM.
    """
    def log_event(self, event_type: str, user_id: Optional[str], room: Optional[str], detail: str):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "room": room,
            "detail": detail
        }
        logger.info(f"AUDIT: {log_entry}")
        # TODO: Persister en base ou envoyer à un SIEM

# Exemple d'utilisation :
# audit_logger = AuditLogger()
# audit_logger.log_event("connect", user_id, room, "Connexion WebSocket acceptée")
