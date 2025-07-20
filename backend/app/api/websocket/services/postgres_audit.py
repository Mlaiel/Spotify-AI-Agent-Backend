import asyncpg
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("PostgresAudit")

class PostgresAuditService:
    """
    Service d'audit PostgreSQL pour journaliser toutes les actions sensibles WebSocket (connexion, message, erreur, etc.).
    """
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(dsn=self.dsn)
        logger.info("Connecté à PostgreSQL pour l'audit WebSocket")

    async def log_event(self, event_type: str, user_id: Optional[str], room: Optional[str], detail: str):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """)
                INSERT INTO websocket_audit (timestamp, event_type, user_id, room, detail)
                VALUES ($1, $2, $3, $4, $5)
                """,
                datetime.utcnow(), event_type, user_id, room, detail
            )
        logger.info(f"Audit PostgreSQL: {event_type} | {user_id} | {room} | {detail}")

    async def close(self):
        await self.pool.close()

# Table SQL recommandée :
# CREATE TABLE websocket_audit (
#   id SERIAL PRIMARY KEY,
#   timestamp TIMESTAMP,
#   event_type TEXT,
#   user_id TEXT,
#   room TEXT,
#   detail TEXT
# );
