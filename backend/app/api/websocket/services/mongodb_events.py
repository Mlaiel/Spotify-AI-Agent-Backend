import motor.motor_asyncio
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("MongoDBEvents")

class MongoDBEventsService:
    """
    Service MongoDB pour stocker les événements WebSocket (analytics, logs, conformité RGPD).
    """
    def __init__(self, mongo_url: str, db_name: str = "websocket", collection: str = "events"):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection]

    async def log_event(self, event_type: str, user_id: Optional[str], room: Optional[str], detail: str):
        doc = {
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "user_id": user_id,
            "room": room,
            "detail": detail
        }
        await self.collection.insert_one(doc)
        logger.info(f"Event MongoDB: {event_type} | {user_id} | {room} | {detail}")

    async def close(self):
        self.client.close()

# Index recommandé :
# db.events.createIndex({"timestamp": -1})
