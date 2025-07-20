import logging
from typing import Dict, Any
from pydantic import BaseModel, Field

class UserDataSyncRequest(BaseModel):
    token: str = Field(..., description="Token d'accès Spotify")

class UserDataSync:
    """
    Synchronisation des données utilisateur Spotify (profils, playlists, stats, logs, sécurité).
    """
    def __init__(self, spotify_client):
        self.logger = logging.getLogger("UserDataSync")
        self.spotify_client = spotify_client

    def sync(self, req: UserDataSyncRequest) -> Dict[str, Any]:
        sp = self.spotify_client.get_client(req.token)
        profile = sp.current_user()
        playlists = sp.current_user_playlists()
        # Ici, on pourrait synchroniser avec la base locale, détecter les changements, etc.
        self.logger.info(f"Données utilisateur synchronisées pour {profile.get('id')}")
        return {"profile": profile, "playlists": playlists["items"]}
