import logging
from typing import Dict, Any
from pydantic import BaseModel, Field

class ArtistInsightsRequest(BaseModel):
    artist_id: str = Field(..., description="ID Spotify de l'artiste")
    token: str = Field(..., description="Token d'accès Spotify")

class ArtistInsights:
    """
    Récupération et analyse des statistiques avancées d'un artiste Spotify (audience, top tracks, analytics, géo, démographie, trends).
    """
    def __init__(self, spotify_client):
        self.logger = logging.getLogger("ArtistInsights")
        self.spotify_client = spotify_client

    def get_insights(self, req: ArtistInsightsRequest) -> Dict[str, Any]:
        sp = self.spotify_client.get_client(req.token)
        artist = sp.artist(req.artist_id)
        top_tracks = sp.artist_top_tracks(req.artist_id)
        related = sp.artist_related_artists(req.artist_id)
        # Simuler analytics avancés (ex: audience géo, trends)
        insights = {
            "artist": artist,
            "top_tracks": top_tracks["tracks"],
            "related_artists": related["artists"],
            "geo_audience": {"FR": 12000, "US": 8000, "DE": 5000},
            "demographics": {"18-24": 40, "25-34": 35, "35-44": 15, "45+": 10},
            "trends": {"monthly_listeners": 50000, "growth": "+8%"}
        }
        self.logger.info(f"Insights récupérés pour artiste {req.artist_id}")
        return insights
