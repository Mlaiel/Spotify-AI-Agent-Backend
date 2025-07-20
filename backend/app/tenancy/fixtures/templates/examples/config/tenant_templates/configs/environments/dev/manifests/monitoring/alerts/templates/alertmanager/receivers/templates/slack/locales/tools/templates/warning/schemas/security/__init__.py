"""
Spotify AI Agent - Tenancy Security Schemas Module
================================================================

Architecture de sécurité multi-tenant avancée pour l'agent IA Spotify.
Ce module fournit une infrastructure complète de sécurité tenant-aware
avec validation, monitoring et alerting en temps réel.

Fonctionnalités principales:
- Validation de schémas de sécurité multi-tenant
- Alerting automatisé avec templates configurables
- Monitoring de sécurité en temps réel
- Intégration Slack pour notifications critiques
- Support multi-langues (fr, en, de, es, it)
- Templates d'alertes personnalisables
- Gestion des permissions par tenant
- Audit trail complet
- Détection d'anomalies comportementales
- Chiffrement différentiel par tenant

Architecture:
- Schemas: Validation et définition des règles de sécurité
- Templates: Modèles d'alertes réutilisables
- Validators: Validateurs de sécurité spécialisés
- Monitors: Surveillance en temps réel
- Processors: Traitement des événements de sécurité
- Integrations: Intégrations externes (Slack, SIEM, etc.)

Auteur: Fahed Mlaiel
Licence: Propriétaire - Tous droits réservés
Version: 1.0.0
Date: 2025-07-19
"""

from .core import (
    SecuritySchemaManager,
    TenantSecurityValidator,
    SecurityEventProcessor,
    AlertingEngine
)

from .schemas import (
    TenantSecuritySchema,
    SecurityRuleSchema,
    AlertConfigSchema,
    PermissionSchema,
    AuditSchema
)

from .validators import (
    TenantAccessValidator,
    PermissionValidator,
    SecurityRuleValidator,
    ComplianceValidator
)

from .monitors import (
    SecurityMonitor,
    ThreatDetector,
    AnomalyDetector,
    ComplianceMonitor
)

from .processors import (
    SecurityEventProcessor,
    AlertProcessor,
    AuditProcessor,
    ThreatProcessor
)

from .integrations import (
    SlackIntegration,
    SIEMIntegration,
    LoggingIntegration,
    MetricsIntegration
)

from .templates import (
    AlertTemplateManager,
    NotificationTemplateManager,
    ReportTemplateManager
)

from .utils import (
    SecurityUtils,
    EncryptionUtils,
    ValidationUtils,
    AuditUtils
)

from .exceptions import (
    SecuritySchemaError,
    TenantSecurityError,
    ValidationError,
    ComplianceError,
    AlertingError
)

__all__ = [
    # Core components
    'SecuritySchemaManager',
    'TenantSecurityValidator',
    'SecurityEventProcessor',
    'AlertingEngine',
    
    # Schemas
    'TenantSecuritySchema',
    'SecurityRuleSchema',
    'AlertConfigSchema',
    'PermissionSchema',
    'AuditSchema',
    
    # Validators
    'TenantAccessValidator',
    'PermissionValidator',
    'SecurityRuleValidator',
    'ComplianceValidator',
    
    # Monitors
    'SecurityMonitor',
    'ThreatDetector',
    'AnomalyDetector',
    'ComplianceMonitor',
    
    # Processors
    'SecurityEventProcessor',
    'AlertProcessor',
    'AuditProcessor',
    'ThreatProcessor',
    
    # Integrations
    'SlackIntegration',
    'SIEMIntegration',
    'LoggingIntegration',
    'MetricsIntegration',
    
    # Templates
    'AlertTemplateManager',
    'NotificationTemplateManager',
    'ReportTemplateManager',
    
    # Utils
    'SecurityUtils',
    'EncryptionUtils',
    'ValidationUtils',
    'AuditUtils',
    
    # Exceptions
    'SecuritySchemaError',
    'TenantSecurityError',
    'ValidationError',
    'ComplianceError',
    'AlertingError'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Fahed Mlaiel"
__email__ = "fahed.mlaiel@achiri.com"
__license__ = "Propriétaire"
__copyright__ = "© 2025 Achiri - Tous droits réservés"
__description__ = "Module de sécurité multi-tenant avancé pour Spotify AI Agent"
__status__ = "Production"
