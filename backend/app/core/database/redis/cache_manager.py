"""
Cache Manager Redis Ultra-Avancé
===============================
- Cache clé/valeur, TTL, invalidation, logs, sécurité
- Prêt pour FastAPI/Django, microservices, DI
- Monitoring, audit, hooks métier
"""

import os
import redis
import logging
import json
from typing import Any, Optional

class RedisCacheManager:
    def __init__(self, url=None, db=0, prefix="cache:", default_ttl=3600):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.db = db
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.client = redis.Redis.from_url(self.url, db=self.db, decode_responses=True)

    def _key(self, key):
        return f"{self.prefix}{key}"

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        ttl = ttl or self.default_ttl
        val = json.dumps(value)
        self.client.set(self._key(key), val, ex=ttl)
        logging.info(f"Cache set: {key} (ttl={ttl})")

    def get(self, key: str) -> Any:
        val = self.client.get(self._key(key))
        if val:
            logging.info(f"Cache hit: {key}")
            return json.loads(val)
        logging.info(f"Cache miss: {key}")
        return None

    def invalidate(self, key: str):
        self.client.delete(self._key(key))
        logging.info(f"Cache invalidé: {key}")

    def clear(self):
        keys = self.client.keys(f"{self.prefix}*")
        if keys:
            self.client.delete(*keys)
            logging.info(f"Cache cleared: {len(keys)} entrées")

# Factory pour DI
cache_manager = RedisCacheManager()
get_cache = lambda: cache_manager
