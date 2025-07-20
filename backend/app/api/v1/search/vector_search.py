import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import numpy as np

class VectorSearchRequest(BaseModel):
    query_vector: List[float] = Field(..., description="Vecteur d'embedding de la requête")
    index_path: str = Field(..., description="Chemin du fichier d'index FAISS")
    top_k: int = Field(10, ge=1, le=100, description="Nombre de résultats à retourner")

class VectorSearch:
    """
    Recherche vectorielle sur index FAISS (ou OpenSearch vectoriel).
    """
    def __init__(self):
        self.logger = logging.getLogger("VectorSearch")

    def search(self, req: VectorSearchRequest) -> List[Dict[str, Any]]:
        import faiss
        import pickle
        # Chargement de l'index FAISS et des métadonnées
        index = faiss.read_index(req.index_path)
        with open(req.index_path + ".meta", "rb") as f:
            meta = pickle.load(f)
        query = np.array([req.query_vector]).astype('float32')
        D, I = index.search(query, req.top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(meta):
                results.append({"id": meta[idx]["id"], "score": float(dist), **meta[idx]})
        self.logger.info(f"Recherche vectorielle sur {req.index_path} (top {req.top_k})")
        return results

# Exemple d'utilisation
# searcher = VectorSearch()
# req = VectorSearchRequest(query_vector=[...], index_path="faiss_index.idx", top_k=5)
# results = searcher.search(req)
