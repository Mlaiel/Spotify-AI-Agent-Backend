"""
Spotify AI Agent – Tasks Root Package
-------------------------------------
Zentraler Einstiegspunkt für alle Task-Submodule:
- Celery-Infrastruktur, Spotify-Tasks, AI/ML, Analytics, Maintenance
- Siehe README für Architektur, Best Practices und Rollen
"""
from .celery import *
from .spotify_tasks import *
from .ai_tasks import *
from .analytics_tasks import *
from .maintenance_tasks import *
