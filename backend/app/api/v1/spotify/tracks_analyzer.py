import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class TracksAnalyzeRequest(BaseModel):
    track_ids: List[str] = Field(..., description="Liste d'IDs de morceaux Spotify")
    token: str = Field(..., description="Token d'accès Spotify")

class TracksAnalyzer:
    """
    Analyse avancée des morceaux Spotify (audio features, mood, genre, recommandations, logs).
    """
    def __init__(self, spotify_client):
        self.logger = logging.getLogger("TracksAnalyzer")
        self.spotify_client = spotify_client

    def analyze_tracks(self, req: TracksAnalyzeRequest) -> List[Dict[str, Any]]:
        sp = self.spotify_client.get_client(req.token)
        features = sp.audio_features(req.track_ids)
        # Simuler mood/genre/reco (en vrai, croiser avec IA interne)
        for f in features:
            f["mood"] = "chill" if f["valence"] > 0.5 else "dark"
            f["genre"] = "lofi" if f["tempo"] < 100 else "edm"
            f["recommendation"] = f["genre"] == "lofi"
        self.logger.info(f"Analyse de {len(req.track_ids)} tracks Spotify")
        return features
