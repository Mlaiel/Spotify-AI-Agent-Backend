import logging
from elasticsearch import exceptions as es_exceptions
from typing import Dict, Any

logger = logging.getLogger("ElasticsearchIndexManager")

class ElasticsearchIndexManager:
    """
    Advanced index manager for Elasticsearch: create, delete, update, mapping, security, monitoring.
    """
    def __init__(self, client):
        self.client = client

    async def create_index(self, index: str, mappings: Dict[str, Any], settings: Dict[str, Any] = None):
        try:
            exists = await self.client.indices.exists(index=index)
            if not exists:
                await self.client.indices.create(index=index, mappings=mappings, settings=settings or {})
                logger.info(f"Index created: {index}")
            else:
                logger.info(f"Index already exists: {index}")
        except es_exceptions.ElasticsearchException as e:
            logger.error(f"Failed to create index {index}: {e}")
            raise

    async def delete_index(self, index: str):
        try:
            await self.client.indices.delete(index=index, ignore=[400, 404])
            logger.info(f"Index deleted: {index}")
        except es_exceptions.ElasticsearchException as e:
            logger.error(f"Failed to delete index {index}: {e}")
            raise

    async def update_mapping(self, index: str, mappings: Dict[str, Any]):
        try:
            await self.client.indices.put_mapping(index=index, body=mappings)
            logger.info(f"Mapping updated for index: {index}")
        except es_exceptions.ElasticsearchException as e:
            logger.error(f"Failed to update mapping for {index}: {e}")
            raise

# Example usage:
# from .client import ElasticsearchClient
# es = ElasticsearchClient(...)
# await es.connect()
# manager = ElasticsearchIndexManager(es.client)
# await manager.create_index("music", mappings={...})
