"""
Spotify Tasks Modul
------------------
Produktionsreife, auditable, sichere und erweiterbare Tasks für Spotify-Integrationen:
- Künstlerüberwachung, Daten-Sync, Playlist-Updates, Streaming-Metriken, Track-Analyse, Social Media Sync, AI Content Generation
- Integriert Security, Audit, Observability, ML/AI-Hooks, GDPR, Prometheus, OpenTelemetry, Sentry

Autoren & Rollen:
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect
"""

from .artist_monitoring import monitor_artist
from .data_sync import sync_artist_data
from .playlist_update import update_playlist
from .streaming_metrics import aggregate_metrics
from .track_analysis import analyze_track
from .social_media_sync import sync_social_media
from .ai_content_generation import generate_content

__all__ = [
    "monitor_artist",
    "sync_artist_data",
    "update_playlist",
    "aggregate_metrics",
    "analyze_track",
    "sync_social_media",
    "generate_content",
]
