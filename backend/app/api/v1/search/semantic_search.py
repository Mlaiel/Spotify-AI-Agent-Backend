import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Requête utilisateur (texte)")
    index: str = Field(..., description="Nom de l'index Elasticsearch ou FAISS")
    size: int = Field(10, ge=1, le=100, description="Nombre de résultats")

class SemanticSearch:
    """
    Recherche sémantique IA (embeddings, transformers HuggingFace, OpenAI, etc.).
    """
    def __init__(self, es_client, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.logger = logging.getLogger("SemanticSearch")
        self.es_client = es_client
        self.embedding_model = embedding_model
        self._load_model()

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.embedding_model)
        self.logger.info(f"Modèle d'embedding chargé: {self.embedding_model}")

    async def search(self, req: SemanticSearchRequest) -> List[Dict[str, Any]]:
        query_vec = self.model.encode([req.query])[0].tolist()
        body = {
            "size": req.size,
            "query": {
                "script_score": {
                    "query": {"match_all": {},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vec}
                    }
                }
            }
        }
        resp = await self.es_client.search(index=req.index, body=body)
        self.logger.info(f"Recherche sémantique sur {req.index} pour '{req.query}'")
        return resp.get("hits", {}).get("hits", [])

# Exemple d'utilisation
# from .elasticsearch_client import ElasticsearchClient
# es = ElasticsearchClient()
# searcher = SemanticSearch(es)
# req = SemanticSearchRequest(query="chill lofi", index="tracks", size=5)
# results = await searcher.search(req)
