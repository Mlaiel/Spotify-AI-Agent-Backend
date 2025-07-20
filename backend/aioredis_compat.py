"""
Compatibility layer for aioredis - Spotify AI Agent
==================================================
Wrapper pour maintenir la compatibilité avec Python 3.13

Auteur: Équipe Lead Dev + Architecte IA
"""

import asyncio
import redis.asyncio as redis
from typing import Any, Optional, Dict, Union, List

# Simulation aioredis avec redis.asyncio
class Redis:
    """Wrapper Redis compatible aioredis"""
    
    def __init__(self, host='localhost', port=6379, db=0, password=None, **kwargs):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._redis = None
        
    async def __aenter__(self):
        self._redis = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._redis:
            await self._redis.aclose()
    
    async def get(self, key: str) -> Optional[str]:
        if not self._redis:
            return None
        return await self._redis.get(key)
    
    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        if not self._redis:
            return False
        return await self._redis.set(key, value, ex=ex)
    
    async def delete(self, *keys) -> int:
        if not self._redis:
            return 0
        return await self._redis.delete(*keys)
    
    async def exists(self, key: str) -> bool:
        if not self._redis:
            return False
        return bool(await self._redis.exists(key))
    
    async def expire(self, key: str, time: int) -> bool:
        if not self._redis:
            return False
        return await self._redis.expire(key, time)
    
    async def ping(self) -> bool:
        if not self._redis:
            return False
        try:
            await self._redis.ping()
            return True
        except:
            return False

# Factory function compatible
def from_url(url: str, **kwargs) -> Redis:
    """Create Redis instance from URL"""
    # Parse simple URL for now
    if url.startswith('redis://'):
        parts = url.replace('redis://', '').split(':')
        host = parts[0] if parts else 'localhost'
        port = int(parts[1]) if len(parts) > 1 else 6379
        return Redis(host=host, port=port, **kwargs)
    return Redis(**kwargs)

# Export compatibility
StrictRedis = Redis
