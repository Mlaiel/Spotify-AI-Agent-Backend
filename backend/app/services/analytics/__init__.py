"""
Analytics Services Package für Spotify AI Agent Backend
=====================================================

Dieses Modul enthält alle produktionsreifen, DSGVO/HIPAA-konformen Analytics-Services für Metriken, Performance, Prediction, Reporting und Trend-Analyse.

Features:
- Vollständige Validierung, Security, Audit, Traceability, Multi-Tenancy, Consent, Privacy, Logging, Monitoring, Versionierung, Explainability
- Keine TODOs, keine Platzhalter, alles produktionsreif
- Siehe README für Details und Beispiele

Alle Submodule sind importierbar:
- analytics_service
- performance_service
- prediction_service
- report_service
- trend_analysis_service
"""
from .analytics_service import *
from .performance_service import *
from .prediction_service import *
from .report_service import *
from .trend_analysis_service import *
