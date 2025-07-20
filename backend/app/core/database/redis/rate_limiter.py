"""
Rate Limiter Redis
=================
- Limitation par token bucket/sliding window, anti-abus, logs
- Sécurité, audit, DI ready
"""

import os
import redis
import time
import logging

class RedisRateLimiter:
    def __init__(self, url=None, db=0, prefix="ratelimit:"):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.db = db
        self.prefix = prefix
        self.client = redis.Redis.from_url(self.url, db=self.db, decode_responses=True)

    def _key(self, key):
        return f"{self.prefix}{key}"

    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        now = int(time.time())
        p = self.client.pipeline()
        p.zadd(self._key(key), {now: now})
        p.zremrangebyscore(self._key(key), 0, now - window)
        p.zcard(self._key(key))
        p.expire(self._key(key), window)
        _, _, count, _ = p.execute()
        allowed = count <= limit
        logging.info(f"Rate limit {key}: {count}/{limit} (allowed={allowed})")
        return allowed

# Factory pour DI
rate_limiter = RedisRateLimiter()
get_rate_limiter = lambda: rate_limiter
