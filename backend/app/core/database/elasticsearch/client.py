import logging
from elasticsearch import AsyncElasticsearch, exceptions as es_exceptions
from typing import Optional

logger = logging.getLogger("ElasticsearchClient")

class ElasticsearchClient:
    """
    Async, secure, production-ready Elasticsearch client for Spotify AI Agent.
    Handles connection pooling, retries, security, logging, and monitoring.
    """
    def __init__(self, hosts: Optional[list] = None, username: Optional[str] = None, password: Optional[str] = None, use_ssl: bool = True, ca_certs: Optional[str] = None):
        self.hosts = hosts or ["http://localhost:9200"]
        self.username = username
        self.password = password
        self.use_ssl = use_ssl
        self.ca_certs = ca_certs
        self.client = None

    async def connect(self):
        self.client = AsyncElasticsearch(
            hosts=self.hosts,
            http_auth=(self.username, self.password) if self.username and self.password else None,
            use_ssl=self.use_ssl,
            ca_certs=self.ca_certs,
            verify_certs=self.use_ssl,
            timeout=30,
            max_retries=5,
            retry_on_timeout=True
        )
        logger.info(f"Connected to Elasticsearch: {self.hosts}")

    async def close(self):
        if self.client:
            await self.client.close()
            logger.info("Elasticsearch connection closed.")

    async def ping(self) -> bool:
        try:
            return await self.client.ping()
        except es_exceptions.ConnectionError as e:
            logger.error(f"Elasticsearch ping failed: {e}")
            return False

# Example usage:
# es = ElasticsearchClient(hosts=["http://localhost:9200"], username="elastic", password="...")
# await es.connect()
# is_alive = await es.ping()
# await es.close()
