"""
Module gRPC industriel pour l’agent IA Spotify.
Expose : AIService, AnalyticsService, MusicService, .proto.
"""

from .ai_service import AIServiceServicer
from .analytics_service import AnalyticsServiceServicer
from .music_service import MusicServiceServicer

__all__ = [
    "AIServiceServicer",
    "AnalyticsServiceServicer",
    "MusicServiceServicer"
]
