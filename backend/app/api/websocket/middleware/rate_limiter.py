import time
import logging
from fastapi import WebSocket, status
from fastapi.exceptions import WebSocketException
from typing import Dict
from collections import defaultdict

logger = logging.getLogger("RateLimiter")

class InMemoryRateLimiter:
    """
    Limiteur de débit WebSocket (par IP/utilisateur). Pour la prod, utiliser Redis !
    """
    def __init__(self, max_messages: int = 20, window_seconds: int = 10):
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self.user_timestamps: Dict[str, list] = defaultdict(list)

    def check(self, user_id: str):
        now = time.time()
        timestamps = self.user_timestamps[user_id]
        # Nettoyage des timestamps expirés
        self.user_timestamps[user_id] = [t for t in timestamps if now - t < self.window_seconds]
        if len(self.user_timestamps[user_id]) >= self.max_messages:
            logger.warning(f"Rate limit dépassé pour {user_id}")
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Trop de messages, ralentissez.")
        self.user_timestamps[user_id].append(now)

# Exemple d'utilisation :
# rate_limiter = InMemoryRateLimiter()
# try:
#     rate_limiter.check(user_id)
# except WebSocketException as e:
#     await websocket.close(code=e.code)
#     raise
