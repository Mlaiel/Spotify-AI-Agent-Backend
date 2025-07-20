import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

class StreamingAnalyticsRequest(BaseModel):
    artist_id: str = Field(..., description="ID Spotify de l'artiste")
    token: str = Field(..., description="Token d'accès Spotify")
    start_date: datetime = Field(..., description="Date de début de la période d'analyse")
    end_date: datetime = Field(..., description="Date de fin de la période d'analyse")

class StreamingAnalytics:
    """
    Analyse avancée des données de streaming Spotify (écoutes, tendances, heatmaps, stats, logs).
    """
    def __init__(self, spotify_client):
        self.logger = logging.getLogger("StreamingAnalytics")
        self.spotify_client = spotify_client

    def get_analytics(self, req: StreamingAnalyticsRequest) -> Dict[str, Any]:
        sp = self.spotify_client.get_client(req.token)
        # Ici, on simule des analytics avancés (en vrai, il faudrait croiser avec des bases internes)
        analytics = {
            "artist_id": req.artist_id,
            "period": {"start": req.start_date.isoformat(), "end": req.end_date.isoformat()},
            "streams": 120000,
            "unique_listeners": 35000,
            "top_countries": {"FR": 40000, "US": 30000, "DE": 20000},
            "hourly_heatmap": [
                {"hour": h, "streams": int(1000 + 500 * (h%6))} for h in range(24)
            ],
            "growth": "+12%"
        }
        self.logger.info(f"Analytics streaming récupérés pour artiste {req.artist_id}")
        return analytics
