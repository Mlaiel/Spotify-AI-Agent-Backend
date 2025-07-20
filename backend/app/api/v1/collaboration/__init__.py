"""
Module Collaboration pour Spotify AI Agent

Ce package expose tous les services de collaboration avancée pour artistes Spotify :
- Workspaces partagés, rooms, notifications, versioning, synchronisation temps réel
- Résolution de conflits, gestion des droits, audit, sécurité
- Intégration scalable (FastAPI, microservices, WebSocket, Redis, MongoDB)

Auteur : Lead Dev, Architecte IA, Backend Senior, Data Engineer, Sécurité
"""

from .shared_workspaces import SharedWorkspaces
from .room_manager import RoomManager
from .notification_system import NotificationSystem
from .conflict_resolution import ConflictResolution
from .version_control import VersionControl
from .real_time_sync import RealTimeSync

__all__ = [
    "SharedWorkspaces",
    "RoomManager",
    "NotificationSystem",
    "ConflictResolution",
    "VersionControl",
    "RealTimeSync",
]
