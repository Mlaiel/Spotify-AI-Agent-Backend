import redis.asyncio as aioredis
import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger("RedisPubSub")

class RedisPubSub:
    """
    Service Pub/Sub Redis pour le broadcast inter-instances (scalabilité WebSocket).
    """
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None

    async def connect(self):
        self.redis = await aioredis.from_url(self.redis_url)
        logger.info("Connecté à Redis pour Pub/Sub")

    async def publish(self, channel: str, message: str):
        await self.redis.publish(channel, message)
        logger.info(f"Message publié sur {channel}")

    async def subscribe(self, channel: str, callback: Callable[str], Any]):
        res = await self.redis.subscribe(channel)
        ch = res[0]
        async for msg in ch.iter(encoding="utf-8"):
            logger.info(f"Message reçu sur {channel}: {msg}")
            await callback(msg)

    async def close(self):
        self.redis.close()
        await self.redis.wait_closed()
