"""
Spotify Services Package

Zentraler Einstiegspunkt für alle Spotify-bezogenen Services:
- API-Integration, Analytics, ML, Playlists, Streaming, User Data
- Automatische Service-Discovery
- Siehe README für Details
"""

from .spotify_api_service import SpotifyAPIService
from .artist_insights_service import ArtistInsightsService
from .playlist_service import PlaylistService
from .streaming_service import StreamingService
from .track_analysis_service import TrackAnalysisService
from .user_data_service import UserDataService

__all__ = [
    "SpotifyAPIService",
    "ArtistInsightsService",
    "PlaylistService",
    "StreamingService",
    "TrackAnalysisService",
    "UserDataService",
]
