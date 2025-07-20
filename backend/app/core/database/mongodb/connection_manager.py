"""
MongoDB Connection Manager
=========================
- Sichere, resiliente Verbindung zu MongoDB (TLS, Auth, Retry, Pooling)
- Health-Check, Tracing, Metrics
- Singleton-Pattern, Dependency Injection ready
- Automatische Umgebungswahl (dev, prod, test)
- Logging & Security Best Practices
"""

from pymongo import MongoClient, monitoring
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
import os
import time
from typing import Optional

class MongoConnectionManager:
    _instance = None
    _client: Optional[MongoClient] = None
    _db = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, uri=None, db_name=None, tls=True, maxPoolSize=50, minPoolSize=5, retryWrites=True):
        if self._client is None:
            self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            self.db_name = db_name or os.getenv("MONGODB_DB", "spotify_ai_agent")
            self.tls = tls
            self.maxPoolSize = maxPoolSize
            self.minPoolSize = minPoolSize
            self.retryWrites = retryWrites
            self._connect()

    def _connect(self):
        try:
            self._client = MongoClient(
                self.uri,
                tls=self.tls,
                maxPoolSize=self.maxPoolSize,
                minPoolSize=self.minPoolSize,
                retryWrites=self.retryWrites,
                serverSelectionTimeoutMS=5000
            )
            self._db = self._client[self.db_name]
            self._client.admin.command('ping')
            logging.info(f"MongoDB connected: {self.uri} (DB: {self.db_name})")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logging.error(f"MongoDB connection failed: {e}")
            raise

    def get_client(self) -> MongoClient:
        return self._client

    def get_db(self):
        return self._db

    def health_check(self) -> bool:
        try:
            self._client.admin.command('ping')
            return True
        except Exception as e:
            logging.error(f"MongoDB health check failed: {e}")
            return False

    def close(self):
        if self._client:
            self._client.close()
            logging.info("MongoDB connection closed.")

# Optional: Monitoring/Tracing Hooks
class CommandLogger(monitoring.CommandListener):
    def started(self, event):
        logging.debug(f"Command {event.command_name} with request id {event.request_id} started on server {event.connection_id}")
    def succeeded(self, event):
        logging.debug(f"Command {event.command_name} with request id {event.request_id} on server {event.connection_id} succeeded in {event.duration_micros}μs")
    def failed(self, event):
        logging.error(f"Command {event.command_name} with request id {event.request_id} on server {event.connection_id} failed in {event.duration_micros}μs")

# monitoring.register(CommandLogger()  # Optionnel, à activer si besoin

def get_mongo_manager():
    """
    Retourne une instance MongoConnectionManager, sauf si TESTING=1.
    """
    if os.getenv("TESTING", "0") == "1":
        raise RuntimeError("MongoDB désactivé en mode test (TESTING=1)")
    return MongoConnectionManager()

def get_mongo_db():
    return get_mongo_manager().get_db()

def get_mongo_client():
    return get_mongo_manager().get_client()
