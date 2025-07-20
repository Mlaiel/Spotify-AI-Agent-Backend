"""
📊 Tenant Models - Modèles de Données Multi-Tenant
==================================================

Modèles de données centralisés pour l'architecture multi-tenant.
Définit les entités principales et leurs relations.

Author: DBA & Data Engineer + Lead Dev - Fahed Mlaiel
Version: 1.0.0
"""

from .tenant import Tenant, TenantSettings, TenantFeature
from .user import TenantUser, UserRole, UserPermission
from .billing import Subscription, Invoice, Payment, UsageRecord
from .analytics import Metric, DataPoint, Dashboard, Report
from .compliance import ConsentRecord, AuditLog, ComplianceReport
from .backup import Backup, RestorePoint, BackupPolicy
from .migration import Migration, MigrationStep, MigrationLog

__all__ = [
    # Core tenant models
    "Tenant",
    "TenantSettings", 
    "TenantFeature",
    
    # User models
    "TenantUser",
    "UserRole",
    "UserPermission",
    
    # Billing models
    "Subscription",
    "Invoice",
    "Payment",
    "UsageRecord",
    
    # Analytics models
    "Metric",
    "DataPoint",
    "Dashboard",
    "Report",
    
    # Compliance models
    "ConsentRecord",
    "AuditLog",
    "ComplianceReport",
    
    # Backup models
    "Backup",
    "RestorePoint",
    "BackupPolicy",
    
    # Migration models
    "Migration",
    "MigrationStep",
    "MigrationLog"
]
