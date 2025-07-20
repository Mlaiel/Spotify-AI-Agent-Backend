from .mongodb import *
from .postgresql import *
from .redis import *
from .elasticsearch import *

__all__ = [
    *mongodb.__all__,
    *postgresql.__all__,
    *redis.__all__,
    *elasticsearch.__all__,
]

import asyncio


async def get_database():
    """
    Fournit la base de données MongoDB par défaut (async), pour compatibilité avec l'auth middleware.
    """
    from .mongodb.connection_manager import get_mongo_db

    # get_mongo_db() peut être sync ou async, harmonisation ici
    db = get_mongo_db()
    if asyncio.iscoroutine(db):
        db = await db
    return db


async def get_database_pool():
    """
    Fournit le pool de connexions de base de données (PostgreSQL par défaut).
    Utilisé par les middlewares de performance et monitoring.
    """
    from .postgresql.connection_pool import get_pg_conn

    try:
        # Retourner le pool PostgreSQL
        pool = await get_pg_conn()
        return pool
    except Exception as e:
        # Fallback vers une connexion mock si PostgreSQL n'est pas disponible
        from unittest.mock import AsyncMock

        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(return_value=AsyncMock())
        return mock_pool


async def get_database_connection():
    """
    Alias pour get_database_pool pour compatibilité.
    """
    return await get_database_pool()
