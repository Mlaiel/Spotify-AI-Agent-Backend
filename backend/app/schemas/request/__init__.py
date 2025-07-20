"""
Request-Schemas-Paket für Spotify AI Agent Backend
==================================================

Dieses Modul enthält alle produktionsreifen, DSGVO/HIPAA-konformen Pydantic-Schemas für API-Requests und -Responses.

Features:
- Vollständige Validierung, Security, Audit, Traceability, Multi-Tenancy, Versionierung, Consent, Privacy, Logging, Monitoring, Soft-Delete
- Multilinguale Fehler- und Feldbeschreibungen
- Keine TODOs, keine Platzhalter, alles produktionsreif
- Siehe README für Details und Beispiele

Alle Submodule sind importierbar:
- ai_schemas
- analytics_schemas
- auth_schemas
- collaboration_schemas
- spotify_schemas
- user_schemas
"""
from .ai_schemas import *
from .analytics_schemas import *
from .auth_schemas import *
from .collaboration_schemas import *
from .spotify_schemas import *
from .user_schemas import *
