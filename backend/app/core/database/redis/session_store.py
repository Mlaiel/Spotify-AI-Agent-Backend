"""
Session Store Redis
==================
- Stockage sessions sécurisé, chiffrement, expiration, audit
- Prêt pour FastAPI/Django, microservices, DI
"""

import os
import redis
import logging
import json
from typing import Any, Optional
from cryptography.fernet import Fernet

class RedisSessionStore:
    def __init__(self, url=None, db=1, prefix="session:", ttl=86400, secret=None):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/1")
        self.db = db
        self.prefix = prefix
        self.ttl = ttl
        self.secret = secret or os.getenv("SESSION_SECRET_KEY", Fernet.generate_key().decode())
        self.fernet = Fernet(self.secret.encode())
        self.client = redis.Redis.from_url(self.url, db=self.db, decode_responses=True)

    def _key(self, session_id):
        return f"{self.prefix}{session_id}"

    def set(self, session_id: str, data: Any):
        val = self.fernet.encrypt(json.dumps(data).encode()).decode()
        self.client.set(self._key(session_id), val, ex=self.ttl)
        logging.info(f"Session set: {session_id}")

    def get(self, session_id: str) -> Optional[Any]:
        val = self.client.get(self._key(session_id))
        if val:
            try:
                data = json.loads(self.fernet.decrypt(val.encode()).decode())
                logging.info(f"Session hit: {session_id}")
                return data
            except Exception as e:
                logging.error(f"Session decrypt failed: {e}")
        logging.info(f"Session miss: {session_id}")
        return None

    def invalidate(self, session_id: str):
        self.client.delete(self._key(session_id))
        logging.info(f"Session invalidée: {session_id}")

    def clear(self):
        keys = self.client.keys(f"{self.prefix}*")
        if keys:
            self.client.delete(*keys)
            logging.info(f"Sessions cleared: {len(keys)} entrées")

# Factory pour DI
session_store = RedisSessionStore()
get_session_store = lambda: session_store
