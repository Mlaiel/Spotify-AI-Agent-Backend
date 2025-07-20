"""
Module de Gestion des Incidents et Métriques - Architecture Industrielle
========================================================================

Ce module fournit une solution complète et industrialisée pour:
- Gestion avancée des incidents de sécurité
- Monitoring des métriques en temps réel
- Alerting intelligent et automatisé
- Forensics et analyse post-incident
- Compliance et audit automatique

Architecture:
    - Event-driven incident management
    - ML-powered anomaly detection  
    - Auto-scaling response systems
    - Real-time analytics dashboard
    - Enterprise-grade security

Auteur: Équipe DevSecOps
Responsable: Fahed Mlaiel
Version: 2.0.0 Enterprise
"""

from .core import (
    IncidentManager,
    MetricsCollector,
    AlertingEngine,
    ForensicsAnalyzer
)

from .handlers import (
    SecurityIncidentHandler,
    PerformanceIncidentHandler,
    BusinessIncidentHandler
)

from .collectors import (
    RealTimeMetricsCollector,
    BusinessMetricsCollector,
    SecurityMetricsCollector
)

from .analyzers import (
    AnomalyDetector,
    TrendAnalyzer,
    PredictiveAnalyzer
)

from .automations import (
    AutoResponseEngine,
    EscalationManager,
    RemediationBot
)

__version__ = "2.0.0"
__enterprise__ = True
__security_grade__ = "A+"

# Configuration par défaut pour l'environnement de développement
DEFAULT_CONFIG = {
    "environment": "development",
    "debug_mode": True,
    "real_time_monitoring": True,
    "auto_remediation": True,
    "ml_anomaly_detection": True,
    "compliance_monitoring": True
}

# Registre des composants principaux
CORE_COMPONENTS = [
    "incident_manager",
    "metrics_collector", 
    "alerting_engine",
    "forensics_analyzer",
    "auto_response_engine"
]

# Niveaux de criticité des incidents
INCIDENT_SEVERITY = {
    "CRITICAL": 1,
    "HIGH": 2, 
    "MEDIUM": 3,
    "LOW": 4,
    "INFO": 5
}

# Types de métriques supportées
METRICS_TYPES = [
    "performance",
    "security",
    "business",
    "infrastructure",
    "application",
    "user_experience"
]
