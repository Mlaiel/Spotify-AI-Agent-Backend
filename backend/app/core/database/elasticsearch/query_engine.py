import logging
from elasticsearch import exceptions as es_exceptions
from typing import Dict, Any, List, Optional

logger = logging.getLogger("ElasticsearchQueryEngine")

class ElasticsearchQueryEngine:
    """
    Advanced query engine for Elasticsearch: fulltext, vector, semantic, filters, security, monitoring.
    """
    def __init__(self, client):
        self.client = client

    async def search(self, index: str, query: Dict[str, Any], size: int = 10, from_: int = 0) -> Dict[str, Any]:
        try:
            response = await self.client.search(index=index, body=query, size=size, from_=from_)
            logger.info(f"Search executed on {index}: {query}")
            return response
        except es_exceptions.ElasticsearchException as e:
            logger.error(f"Search failed on {index}: {e}")
            raise

    async def vector_search(self, index: str, vector: List[float], field: str = "embedding", size: int = 10) -> Dict[str, Any]:
        query = {
            "knn": {
                field: {
                    "vector": vector,
                    "k": size
                }
            }
        }
        return await self.search(index=index, query=query, size=size)

    async def filter_search(self, index: str, filters: Dict[str, Any], size: int = 10) -> Dict[str, Any]:
        query = {"query": {"bool": {"filter": filters}}}
        return await self.search(index=index, query=query, size=size)

# Example usage:
# from .client import ElasticsearchClient
# es = ElasticsearchClient(...)
# await es.connect()
# engine = ElasticsearchQueryEngine(es.client)
# await engine.search("music", query={...})
