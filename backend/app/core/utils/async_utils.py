"""
Module: async_utils.py
Description: Utilitaires avancés pour la gestion de l'asynchrone (timeout, retry, concurrency, FastAPI, asyncio).
"""
import asyncio
from functools import wraps
from typing import Callable, Any, Coroutine

async def async_retry(func: Callable[..., Coroutine], retries: int = 3, delay: float = 1.0) -> Any:
    for attempt in range(retries):
        try:
            return await func()
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)

async def async_timeout(coro: Coroutine, timeout: float) -> Any:
    return await asyncio.wait_for(coro, timeout=timeout)

def run_sync_in_executor(func: Callable, *args, **kwargs) -> Any:
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args, **kwargs)

# Exemples d'utilisation
# await async_retry(lambda: my_async_func(), retries=5)
# await async_timeout(my_async_func(), timeout=2.0)
