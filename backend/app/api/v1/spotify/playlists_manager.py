import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class PlaylistsSyncRequest(BaseModel):
    user_id: str = Field(..., description="ID utilisateur Spotify")
    token: str = Field(..., description="Token d'accès Spotify")

class PlaylistsManager:
    """
    Gestion, synchronisation et analyse des playlists Spotify (création, update, sync, stats, logs).
    """
    def __init__(self, spotify_client):
        self.logger = logging.getLogger("PlaylistsManager")
        self.spotify_client = spotify_client

    def sync_playlists(self, req: PlaylistsSyncRequest) -> List[Dict[str, Any]]:
        sp = self.spotify_client.get_client(req.token)
        playlists = sp.current_user_playlists()
        # Ici, on pourrait synchroniser avec la base locale, détecter les changements, etc.
        self.logger.info(f"Playlists synchronisées pour utilisateur {req.user_id}")
        return playlists["items"]

    def create_playlist(self, req: PlaylistsSyncRequest, name: str, description: str = "") -> Dict[str, Any]:
        sp = self.spotify_client.get_client(req.token)
        playlist = sp.user_playlist_create(req.user_id, name, description=description)
        self.logger.info(f"Playlist créée: {name} pour utilisateur {req.user_id}")
        return playlist
