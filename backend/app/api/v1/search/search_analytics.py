import logging
from typing import Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

class SearchLogEntry(BaseModel):
    user_id: str
    query: str
    timestamp: datetime
    result_count: int
    latency_ms: int
    ip: str

class SearchAnalytics:
    """
    Analytics, audit et monitoring des recherches (logs, stats, heatmaps, top requêtes, RGPD).
    """
    def __init__(self):
        self.logger = logging.getLogger("SearchAnalytics")
        self.logs: List[SearchLogEntry] = []

    def log_search(self, entry: SearchLogEntry):
        self.logs.append(entry)
        self.logger.info(f"Recherche loggée: {entry.user_id} - '{entry.query}' ({entry.result_count} résultats)")

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.logs)
        top_queries = {}
        for log in self.logs:
            top_queries[log.query] = top_queries.get(log.query, 0) + 1
        sorted_queries = sorted(top_queries.items(), key=lambda x: x[1], reverse=True)
        return {
            "total_searches": total,
            "top_queries": sorted_queries[:10],
            "avg_latency_ms": sum(l.latency_ms for l in self.logs)/total if total else 0
        }

    def get_user_history(self, user_id: str) -> List[SearchLogEntry]:
        return [l for l in self.logs if l.user_id == user_id]

# Exemple d'utilisation
# analytics = SearchAnalytics()
# analytics.log_search(SearchLogEntry(user_id="u1", query="chill", timestamp=datetime.now(), result_count=12, latency_ms=120, ip="1.2.3.4")
# stats = analytics.get_stats()
