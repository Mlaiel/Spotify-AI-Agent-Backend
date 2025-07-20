"""
Pub/Sub Manager Redis
====================
- Gestion des channels, events, hooks, logs
- Sécurité, audit, DI ready
"""

import os
import redis
import logging
from threading import Thread

class RedisPubSubManager:
    def __init__(self, url=None, db=0):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.db = db
        self.client = redis.Redis.from_url(self.url, db=self.db, decode_responses=True)
        self.pubsub = self.client.pubsub()

    def publish(self, channel: str, message: str):
        self.client.publish(channel, message)
        logging.info(f"Message publié sur {channel}: {message}")

    def subscribe(self, channel: str, callback):
        def listen():
            self.pubsub.subscribe(channel)
            for msg in self.pubsub.listen():
                if msg['type'] == 'message':
                    logging.info(f"Message reçu sur {channel}: {msg['data']}")
                    callback(msg['data'])
        Thread(target=listen, daemon=True).start()

# Factory pour DI
pubsub_manager = RedisPubSubManager()
get_pubsub = lambda: pubsub_manager
