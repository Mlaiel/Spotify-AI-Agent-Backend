"""
Backend Services Package

Zentraler Einstiegspunkt für alle Backend-Services:
- AI, Analytics, Auth, Cache, Collaboration, Email, Queue, Search, Spotify, Storage
- Siehe README für Details und Submodule
"""
from .ai import *
from .analytics import *
from .auth import *
from .cache import *
from .collaboration import *
from .email import *
from .queue import *
from .search import *
from .spotify import *
from .storage import *

__all__ = [
    "ai", "analytics", "auth", "cache", "collaboration", "email", "queue", "search", "spotify", "storage"
]
