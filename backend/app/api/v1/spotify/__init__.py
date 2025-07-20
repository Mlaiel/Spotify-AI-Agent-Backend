"""
Module d'intégration avancée Spotify pour artistes.
Expose : stats, playlists, webhooks, synchronisation, analytics, analyse tracks.
"""

from .spotify_client import SpotifyClient
from .artist_insights import ArtistInsights
from .playlists_manager import PlaylistsManager
from .spotify_webhook import SpotifyWebhook
from .streaming_analytics import StreamingAnalytics
from .tracks_analyzer import TracksAnalyzer
from .user_data_sync import UserDataSync

__all__ = [
    "SpotifyClient",
    "ArtistInsights",
    "PlaylistsManager",
    "SpotifyWebhook",
    "StreamingAnalytics",
    "TracksAnalyzer",
    "UserDataSync"
]
