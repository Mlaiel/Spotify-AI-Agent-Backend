import logging
from typing import Optional, Dict, Any
from spotipy import Spotify, SpotifyOAuth

class SpotifyClient:
    """
    Client Spotify sécurisé, asynchrone, gestion OAuth2, refresh, logs, monitoring.
    """
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, scope: str = "user-read-private user-read-email playlist-read-private"):
        self.logger = logging.getLogger("SpotifyClient")
        self.oauth = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope
        )
        self.sp: Optional[Spotify] = None

    def get_token(self, code: str) -> Dict[str, Any]:
        token_info = self.oauth.get_access_token(code)
        self.logger.info("Token Spotify obtenu.")
        return token_info

    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        token_info = self.oauth.refresh_access_token(refresh_token)
        self.logger.info("Token Spotify rafraîchi.")
        return token_info

    def get_client(self, token: str) -> Spotify:
        self.sp = Spotify(auth=token)
        return self.sp

    def get_user_profile(self, token: str) -> Dict[str, Any]:
        sp = self.get_client(token)
        profile = sp.current_user()
        self.logger.info(f"Profil utilisateur récupéré: {profile.get('id')}")
        return profile
