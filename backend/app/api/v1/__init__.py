"""
API v1 du backend Spotify AI Agent.
Expose tous les modules : auth, spotify, ai_agent, content_generation, music_generation, search, analytics, collaboration.
"""

# Import public API de chaque sous-module
# (Les routers FastAPI sont à importer dans main.py)

__all__ = [
    "auth", "spotify", "ai_agent", "content_generation", "music_generation", "search", "analytics", "collaboration"
]
