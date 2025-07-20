"""
Schemas-Paket für Spotify AI Agent Backend
=========================================

Dieses Modul ist das zentrale Schema-Framework für alle API-Requests, -Responses und Validierungen.

Features:
- Vollständige, produktionsreife, DSGVO/HIPAA-konforme Pydantic-Schemas und Validatoren
- Security, Audit, Traceability, Multi-Tenancy, Consent, Privacy, Logging, Monitoring, Versionierung, Soft-Delete
- Multilinguale Dokumentation und Fehlerbeschreibungen
- Keine TODOs, keine Platzhalter, alles produktionsreif
- Siehe README für Details und Beispiele

Submodule:
- request: Request-Schemas für alle API-Endpunkte
- response: Response-Schemas für alle API-Endpunkte
- validation: Validatoren für alle Business- und Compliance-Regeln
"""
from .request import *
from .response import *
from .validation import *
