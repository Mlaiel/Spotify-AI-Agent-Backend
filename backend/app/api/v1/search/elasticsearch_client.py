import logging
from typing import Optional, Dict, Any
from elasticsearch import AsyncElasticsearch, NotFoundError

class ElasticsearchClient:
    """
    Client Elasticsearch/OpenSearch sécurisé, asynchrone, multi-index, avec logs et monitoring.
    """
    def __init__(self, hosts: Optional[list] = None, username: Optional[str] = None, password: Optional[str] = None):
        self.logger = logging.getLogger("ElasticsearchClient")
        self.hosts = hosts or ["http://localhost:9200"]
        self.username = username
        self.password = password
        self.client = AsyncElasticsearch(
            hosts=self.hosts,
            http_auth=(self.username, self.password) if self.username and self.password else None,
            max_retries=3,
            retry_on_timeout=True
        )

    async def search(self, index: str, body: Dict[str, Any]) -> Dict:
        try:
            resp = await self.client.search(index=index, body=body)
            self.logger.info(f"Recherche Elasticsearch sur {index} réussie.")
            return resp
        except NotFoundError:
            self.logger.warning(f"Index {index} introuvable.")
            return {"hits": {"hits": []}
        except Exception as e:
            self.logger.error(f"Erreur recherche Elasticsearch: {e}")
            raise

    async def index_document(self, index: str, doc: Dict[str, Any], doc_id: Optional[str] = None):
        try:
            resp = await self.client.index(index=index, id=doc_id, document=doc)
            self.logger.info(f"Document indexé dans {index} (id={doc_id})")
            return resp
        except Exception as e:
            self.logger.error(f"Erreur indexation: {e}")
            raise

    async def create_index(self, index: str, settings: Optional[Dict] = None, mappings: Optional[Dict] = None):
        try:
            exists = await self.client.indices.exists(index=index)
            if not exists:
                await self.client.indices.create(index=index, settings=settings, mappings=mappings)
                self.logger.info(f"Index {index} créé.")
        except Exception as e:
            self.logger.error(f"Erreur création index: {e}")
            raise

    async def close(self):
        await self.client.close()
