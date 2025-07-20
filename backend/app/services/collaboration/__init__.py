"""
Spotify AI Agent – Collaboration Module

Created by: Achiri AI Engineering Team
Roles: Lead Dev + Architecte IA, Développeur Backend Senior, Ingénieur ML, DBA/Data Engineer, Spécialiste Sécurité, Architecte Microservices
"""
from .notification_service import NotificationService
from .permission_service import PermissionService
from .real_time_service import RealTimeService
from .version_control_service import VersionControlService
from .workspace_service import WorkspaceService

__version__ = "1.0.0"
__all__ = [
    "NotificationService",
    "PermissionService",
    "RealTimeService",
    "VersionControlService",
    "WorkspaceService",
]
