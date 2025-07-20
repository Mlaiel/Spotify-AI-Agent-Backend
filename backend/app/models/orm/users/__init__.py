"""
User ORM Models for Spotify AI Agent

Lead Dev: [Name], Architecte IA: [Name], Backend: [Name], ML: [Name], DBA: [Name], Security: [Name], Microservices: [Name]

Dieses Modul enthält alle produktionsreifen, auditierbaren, DSGVO-konformen ORM-Modelle für User-Daten:
- User, UserProfile, UserPreferences, UserSpotifyData, UserSubscription
- Optimiert für Analytics, KI, Recommendation, Monetarisierung, Data Lineage, Multi-Tenancy

Features:
- Vollständige Validierung, Security, Audit, Soft-Delete, Timestamps, User-Attribution, Multi-Tenancy
- CI/CD-ready, Governance, Compliance, Logging, Monitoring, Data Lineage
- Erweiterbar für neue User-Modelle, Pipelines, Integrationen

Alle Modelle sind für PostgreSQL, MongoDB und hybride Architekturen optimiert.

"""

from .user import *
from .user_profile import *
from .user_preferences import *
from .user_spotify_data import *
from .user_subscription import *
