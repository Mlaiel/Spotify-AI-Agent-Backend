"""
AI ORM Models for Spotify AI Agent

Lead Dev: [Name], Architecte IA: [Name], Backend: [Name], ML: [Name], DBA: [Name], Security: [Name], Microservices: [Name]

Dieses Modul enthält alle produktionsreifen, auditierbaren, DSGVO-konformen ORM-Modelle für KI-Features:
- KI-Konversationen (Chat, Prompt, Verlauf, User-Attribution, Multi-Tenancy)
- Feedback & Bewertung (User, Model, Audit, Explainability)
- Generierte Inhalte (Text, Audio, Metadaten, Versionierung, Traceability)
- Modell-Konfiguration (Hyperparameter, Registry, Version, Audit, Security)
- Performance-Metriken (Accuracy, Fairness, Drift, Monitoring, Logging)
- Trainingsdaten (Lineage, Source, Compliance, Audit, Data Quality)

Features:
- Vollständige Validierung, Security, Audit, Soft-Delete, Timestamps, User-Attribution, Multi-Tenancy
- CI/CD-ready, Governance, Compliance, Logging, Explainability, Monitoring, Data Lineage
- Erweiterbar für neue KI-Modelle, Pipelines, Integrationen

Alle Modelle sind für PostgreSQL, MongoDB und hybride Architekturen optimiert.

"""

from .ai_conversation import *
from .ai_feedback import *
from .ai_generated_content import *
from .ai_model_config import *
from .model_performance import *
from .training_data import *
