"""
Response-Schemas-Paket für Spotify AI Agent Backend
==================================================

Dieses Modul enthält alle produktionsreifen, DSGVO/HIPAA-konformen Pydantic-Schemas für API-Responses.

Features:
- Vollständige Validierung, Security, Audit, Traceability, Multi-Tenancy, Versionierung, Consent, Privacy, Logging, Monitoring, Soft-Delete
- Multilinguale Fehler- und Feldbeschreibungen
- Keine TODOs, keine Platzhalter, alles produktionsreif
- Siehe README für Details und Beispiele

Alle Submodule sind importierbar:
- base_response
- ai_response
- analytics_response
- collaboration_response
- spotify_response
- user_response
"""
from .base_response import *
from .ai_response import *
from .analytics_response import *
from .collaboration_response import *
from .spotify_response import *
from .user_response import *
