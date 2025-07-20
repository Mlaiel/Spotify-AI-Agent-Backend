"""
Spotify AI Agent – Backend App Root
----------------------------------
Zentraler Einstiegspunkt für das gesamte Backend:
- API, Core, Models, Services, Tasks, ML, Utils, Migrations, Enums, Schemas, Docs
- Siehe README für Architektur, Best Practices und Rollen
"""

# from .api import *  # ENTFERNT, um zirkuläre Importe zu vermeiden
from .core import *
from .models import *
from .services import *
from .tasks import *
from .ml import *
from .utils import *
from .migrations import *
from .enums import *
from .schemas import *
from .docs import *

__all__ = [
    # "api",  # ENTFERNT, um zirkuläre Importe zu vermeiden
    "core", "models", "services", "tasks", "ml", "utils", "migrations", "enums", "schemas", "docs"
]
