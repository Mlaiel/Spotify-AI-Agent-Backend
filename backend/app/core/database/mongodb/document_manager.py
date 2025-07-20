"""
MongoDB Document Manager
=======================
- CRUD, Validierung, Transaktionen, Versionierung, Soft-Delete, Auditing
- Business-Logik für Spotify AI Agent (z.B. User, Analytics, Content)
- Exception Handling, Logging, Security
- Erweiterbar für komplexe Modelle
"""

from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from bson import ObjectId
import logging
import datetime
from .connection_manager import get_mongo_db

class DocumentManager:
    def __init__(self, collection_name: str):
        self.db = get_mongo_db()
        self.collection: Collection = self.db[collection_name]

    def create(self, data: dict) -> str:
        data['created_at'] = datetime.datetime.utcnow()
        data['updated_at'] = datetime.datetime.utcnow()
        data['deleted'] = False
        try:
            result = self.collection.insert_one(data)
            logging.info(f"Document created in {self.collection.name}: {result.inserted_id}")
            return str(result.inserted_id)
        except PyMongoError as e:
            logging.error(f"Create failed: {e}")
            raise

    def get(self, doc_id: str) -> dict:
        try:
            doc = self.collection.find_one({"_id": ObjectId(doc_id), "deleted": False})
            return doc
        except PyMongoError as e:
            logging.error(f"Get failed: {e}")
            raise

    def update(self, doc_id: str, updates: dict) -> bool:
        updates['updated_at'] = datetime.datetime.utcnow()
        try:
            result = self.collection.update_one({"_id": ObjectId(doc_id), "deleted": False}, {"$set": updates})
            return result.modified_count > 0
        except PyMongoError as e:
            logging.error(f"Update failed: {e}")
            raise

    def delete(self, doc_id: str, soft=True) -> bool:
        try:
            if soft:
                result = self.collection.update_one({"_id": ObjectId(doc_id)}, {"$set": {"deleted": True, "deleted_at": datetime.datetime.utcnow()}})
            else:
                result = self.collection.delete_one({"_id": ObjectId(doc_id)})
            return result.modified_count > 0 or result.deleted_count > 0
        except PyMongoError as e:
            logging.error(f"Delete failed: {e}")
            raise

    def find(self, query: dict = None, limit: int = 100):
        query = query or {"deleted": False}
        try:
            return list(self.collection.find(query).limit(limit))
        except PyMongoError as e:
            logging.error(f"Find failed: {e}")
            raise

    def start_transaction(self):
        return self.db.client.start_session()

    # Beispiel für Versionierung (History-Collection)
    def save_version(self, doc_id: str):
        doc = self.get(doc_id)
        if doc:
            version_col = self.db[f"{self.collection.name}_history"]
            doc['versioned_at'] = datetime.datetime.utcnow()
            version_col.insert_one(doc)

# Beispiel für Business-Logik: UserManager, AnalyticsManager etc. können von DocumentManager erben
