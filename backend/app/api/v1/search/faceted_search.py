import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class FacetedSearchRequest(BaseModel):
    query: str = Field(..., description="Requête de recherche texte")
    facets: List[str] = Field(..., description="Liste des facettes à retourner (ex: genre, mood, pays)")
    index: str = Field(..., description="Nom de l'index Elasticsearch")
    size: int = Field(10, ge=1, le=100, description="Nombre de résultats")

class FacetedSearch:
    """
    Recherche à facettes sur Elasticsearch (genre, mood, pays, etc.).
    """
    def __init__(self, es_client):
        self.logger = logging.getLogger("FacetedSearch")
        self.es_client = es_client

    async def search(self, req: FacetedSearchRequest) -> Dict[str, Any]:
        body = {
            "query": {"multi_match": {"query": req.query, "fields": ["title^2", "description", "tags"]}},
            "aggs": {facet: {"terms": {"field": f"{facet}.keyword"}} for facet in req.facets},
            "size": req.size
        }
        resp = await self.es_client.search(index=req.index, body=body)
        self.logger.info(f"Recherche à facettes sur {req.index} pour '{req.query}'")
        return resp

# Exemple d'utilisation
# from .elasticsearch_client import ElasticsearchClient
# es = ElasticsearchClient()
# searcher = FacetedSearch(es)
# req = FacetedSearchRequest(query="chill", facets=["genre", "mood"], index="tracks", size=10)
# results = await searcher.search(req)
