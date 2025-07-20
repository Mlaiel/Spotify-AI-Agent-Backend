"""
Schémas de configuration pour l'architecture multi-tenant.
Validation et définition des structures de données.
"""

from .tenant_schema import TenantConfigSchema, TenantMetadataSchema
from .monitoring_schema import MonitoringConfigSchema, AlertingSchema
from .security_schema import SecurityPolicySchema, EncryptionSchema
from .environment_schema import EnvironmentConfigSchema
from .localization_schema import LocalizationSchema

__all__ = [
    'TenantConfigSchema',
    'TenantMetadataSchema',
    'MonitoringConfigSchema',
    'AlertingSchema',
    'SecurityPolicySchema',
    'EncryptionSchema',
    'EnvironmentConfigSchema',
    'LocalizationSchema'
]
