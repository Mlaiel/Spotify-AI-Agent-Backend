"""
🏢 Tenancy Managers Module - Spotify AI Agent
==========================================

Gestionnaires avancés pour la gestion multi-tenant.
Architecture industrielle avec patterns enterprise.

Managers disponibles:
- TenantManager: Gestion centrale des tenants
- TenantDataManager: Isolation et gestion des données
- TenantSecurityManager: Sécurité et authentification
- TenantBillingManager: Facturation et quotas
- TenantAnalyticsManager: Métriques et analytics
- TenantComplianceManager: Conformité réglementaire
- TenantMigrationManager: Migrations et déploiements
- TenantProvisioningManager: Provisioning automatisé
- TenantBackupManager: Sauvegardes et restauration
- TenantPerformanceManager: Optimisation performances

Architects: Lead Dev + Architecte IA, DBA & Data Engineer
Version: 1.0.0
"""

from .tenant_manager import TenantManager
from .data_manager import TenantDataManager
from .security_manager import TenantSecurityManager
from .billing_manager import TenantBillingManager
from .analytics_manager import TenantAnalyticsManager
from .compliance_manager import TenantComplianceManager
from .migration_manager import TenantMigrationManager
from .provisioning_manager import TenantProvisioningManager
from .backup_manager import TenantBackupManager
from .performance_manager import TenantPerformanceManager

__all__ = [
    "TenantManager",
    "TenantDataManager", 
    "TenantSecurityManager",
    "TenantBillingManager",
    "TenantAnalyticsManager",
    "TenantComplianceManager",
    "TenantMigrationManager",
    "TenantProvisioningManager",
    "TenantBackupManager",
    "TenantPerformanceManager"
]
