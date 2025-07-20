"""
Spotify AI Agent - Module Tenancy Schemas
========================================

Module de gestion des schémas pour le système multi-tenant avancé.
Ce module contient les définitions de schémas pour la validation,
la configuration et la gestion des alertes tenant-aware.

Architecture:
- Validation des données multi-tenant
- Schémas de configuration per-tenant
- Templates d'alertes personnalisées
- Gestion des warnings et notifications

Version: 1.0.0
Compatibilité: Python 3.11+
"""

__version__ = "1.0.0"
__author__ = "Spotify AI Agent Team"
__email__ = "dev-team@spotify-ai-agent.com"

# Configuration du module
TENANCY_SCHEMA_VERSION = "v1"
SUPPORTED_TENANT_TYPES = ["enterprise", "professional", "standard", "trial"]
DEFAULT_LOCALE = "en"

# Importations principales
from .tenant_config_schema import TenantConfigSchema
from .alert_schema import AlertSchema, TenantAlertSchema
from .warning_schema import WarningSchema, TenantWarningSchema
from .notification_schema import NotificationSchema
from .monitoring_schema import MonitoringConfigSchema
from .compliance_schema import ComplianceSchema
from .performance_schema import PerformanceMetricsSchema

__all__ = [
    "TenantConfigSchema",
    "AlertSchema",
    "TenantAlertSchema", 
    "WarningSchema",
    "TenantWarningSchema",
    "NotificationSchema",
    "MonitoringConfigSchema",
    "ComplianceSchema",
    "PerformanceMetricsSchema",
    "TENANCY_SCHEMA_VERSION",
    "SUPPORTED_TENANT_TYPES",
    "DEFAULT_LOCALE"
]
