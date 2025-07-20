"""
Spotify ORM Models for Spotify AI Agent

Lead Dev: [Name], Architecte IA: [Name], Backend: [Name], ML: [Name], DBA: [Name], Security: [Name], Microservices: [Name]

Dieses Modul enthält alle produktionsreifen, auditierbaren, DSGVO-konformen ORM-Modelle für Spotify-Daten:
- Album, Artist, Track, Audio Features, Genre, Playlist, Streaming Data
- Optimiert für Analytics, KI, Recommendation, Monetarisierung, Data Lineage, Multi-Tenancy

Features:
- Vollständige Validierung, Security, Audit, Soft-Delete, Timestamps, User-Attribution, Multi-Tenancy
- CI/CD-ready, Governance, Compliance, Logging, Monitoring, Data Lineage
- Erweiterbar für neue Spotify-Modelle, Pipelines, Integrationen

Alle Modelle sind für PostgreSQL, MongoDB und hybride Architekturen optimiert.

"""

from .album import *
from .artist import *
from .track import *
from .audio_features import *
from .genre import *
from .playlist import *
from .streaming_data import *
