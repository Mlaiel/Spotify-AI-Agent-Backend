"""
Validation-Schemas-Paket für Spotify AI Agent Backend
====================================================

Dieses Modul enthält alle produktionsreifen, DSGVO/HIPAA-konformen Validatoren für API-Schemas und Business-Logik.

Features:
- Vollständige Validierung, Security, Audit, Traceability, Multi-Tenancy, Versionierung, Consent, Privacy, Logging, Monitoring, Soft-Delete
- Multilinguale Fehler- und Feldbeschreibungen
- Keine TODOs, keine Platzhalter, alles produktionsreif
- Siehe README für Details und Beispiele

Alle Submodule sind importierbar:
- common_validators
- ai_validators
- spotify_validators
- custom_validators
"""
from .common_validators import *
from .ai_validators import *
from .spotify_validators import *
from .custom_validators import *
