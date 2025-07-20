from pydantic import BaseModel, Field, SecretStr, AnyUrl
from typing import List

class SpotifyConfig(BaseModel):
    client_id: str = Field(..., description="Spotify API client ID")
    client_secret: SecretStr = Field(..., description="Spotify API client secret")
    redirect_uri: AnyUrl = Field(..., description="OAuth2 redirect URI")
    scopes: List[str] = Field(default_factory=lambda: [
        "user-read-private", "user-read-email", "playlist-read-private", "playlist-modify-public"
    ], description="OAuth2 scopes")
    auth_url: AnyUrl = Field("https://accounts.spotify.com/authorize", description="Spotify auth URL")
    token_url: AnyUrl = Field("https://accounts.spotify.com/api/token", description="Spotify token URL")

# Example usage:
# from .spotify_config import SpotifyConfig
# spotify_conf = SpotifyConfig(client_id="...", client_secret="...", redirect_uri="http://localhost:8000/callback")
