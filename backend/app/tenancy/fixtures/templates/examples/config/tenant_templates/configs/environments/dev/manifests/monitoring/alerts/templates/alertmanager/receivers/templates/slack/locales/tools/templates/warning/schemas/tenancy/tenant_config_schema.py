"""
Tenant Configuration Schema Module
=================================

Ce module définit les schémas de validation pour la configuration des tenants
dans un environnement multi-tenant industriel avec support avancé des SLAs,
compliance et monitoring.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.networks import HttpUrl, IPvAnyAddress


class TenantType(str, Enum):
    """Types de tenant supportés avec leurs niveaux de service."""
    ENTERPRISE = "enterprise"
    PROFESSIONAL = "professional"
    STANDARD = "standard"
    TRIAL = "trial"


class TenantStatus(str, Enum):
    """États possibles d'un tenant."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PROVISIONING = "provisioning"
    DEPROVISIONING = "deprovisioning"
    MAINTENANCE = "maintenance"
    ARCHIVED = "archived"


class ComplianceLevel(str, Enum):
    """Niveaux de compliance supportés."""
    BASIC = "basic"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"


class IsolationLevel(str, Enum):
    """Niveaux d'isolation des données."""
    STRICT = "strict"
    MODERATE = "moderate"
    BASIC = "basic"


class TenantFeatures(BaseModel):
    """Configuration des fonctionnalités tenant."""
    advanced_analytics: bool = Field(False, description="Analytics avancées")
    custom_alerts: bool = Field(False, description="Alertes personnalisées")
    real_time_monitoring: bool = Field(True, description="Monitoring temps réel")
    api_rate_limiting: bool = Field(True, description="Rate limiting API")
    data_encryption: bool = Field(True, description="Chiffrement des données")
    audit_logging: bool = Field(True, description="Logs d'audit")
    backup_retention: int = Field(30, ge=1, le=365, description="Rétention backup (jours)")
    max_users: Optional[int] = Field(None, ge=1, description="Nombre max d'utilisateurs")
    max_storage_gb: Optional[int] = Field(None, ge=1, description="Stockage max (GB)")
    max_api_calls_per_hour: int = Field(1000, ge=100, description="Limite API/heure")
    priority_support: bool = Field(False, description="Support prioritaire")
    custom_branding: bool = Field(False, description="Branding personnalisé")
    sso_integration: bool = Field(False, description="Intégration SSO")
    multi_region_backup: bool = Field(False, description="Backup multi-région")


class TenantSLA(BaseModel):
    """Définition des SLA pour un tenant."""
    uptime_percentage: float = Field(99.9, ge=95.0, le=100.0, description="Uptime garanti %")
    response_time_ms: int = Field(500, ge=50, le=5000, description="Temps de réponse max (ms)")
    recovery_time_hours: int = Field(4, ge=1, le=72, description="Temps de récupération max (h)")
    backup_frequency_hours: int = Field(24, ge=1, le=168, description="Fréquence backup (h)")
    support_response_minutes: int = Field(60, ge=5, le=1440, description="Réponse support (min)")
    data_retention_days: int = Field(365, ge=30, le=2555, description="Rétention données (jours)")


class TenantSecurityConfig(BaseModel):
    """Configuration de sécurité tenant."""
    encryption_level: str = Field("AES-256", description="Niveau de chiffrement")
    ssl_required: bool = Field(True, description="SSL obligatoire")
    mfa_required: bool = Field(False, description="MFA obligatoire")
    password_policy: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_special": True,
            "max_age_days": 90
        },
        description="Politique de mot de passe"
    )
    session_timeout_minutes: int = Field(60, ge=5, le=480, description="Timeout session (min)")
    ip_whitelist: Optional[List[IPvAnyAddress]] = Field(None, description="Liste IP autorisées")
    allowed_domains: Optional[List[str]] = Field(None, description="Domaines autorisés")
    audit_retention_days: int = Field(365, ge=90, le=2555, description="Rétention audit (jours)")


class TenantMonitoringConfig(BaseModel):
    """Configuration du monitoring tenant."""
    metrics_enabled: bool = Field(True, description="Métriques activées")
    alerting_enabled: bool = Field(True, description="Alerting activé")
    log_level: str = Field("INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    retention_days: int = Field(90, ge=7, le=365, description="Rétention logs (jours)")
    custom_dashboards: bool = Field(False, description="Dashboards personnalisés")
    real_time_alerts: bool = Field(True, description="Alertes temps réel")
    performance_profiling: bool = Field(False, description="Profiling performance")
    error_tracking: bool = Field(True, description="Suivi des erreurs")


class TenantBillingConfig(BaseModel):
    """Configuration de facturation tenant."""
    billing_cycle: str = Field("monthly", regex="^(monthly|yearly|usage_based)$")
    currency: str = Field("USD", regex="^[A-Z]{3}$", description="Code devise ISO")
    auto_scaling_enabled: bool = Field(False, description="Auto-scaling activé")
    overage_charges: bool = Field(True, description="Frais de dépassement")
    payment_method: Optional[str] = Field(None, description="Méthode de paiement")
    credit_limit: Optional[float] = Field(None, ge=0, description="Limite de crédit")
    invoice_email: Optional[str] = Field(None, description="Email facturation")


class TenantConfigSchema(BaseModel):
    """
    Schéma principal de configuration tenant avec validation complète
    et support des fonctionnalités industrielles avancées.
    """
    # Identifiants
    tenant_id: str = Field(..., regex="^[a-zA-Z0-9_-]+$", min_length=3, max_length=64)
    tenant_name: str = Field(..., min_length=2, max_length=100)
    tenant_type: TenantType = Field(..., description="Type de tenant")
    
    # État et métadonnées
    status: TenantStatus = Field(TenantStatus.PROVISIONING, description="État du tenant")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Configuration principale
    features: TenantFeatures = Field(default_factory=TenantFeatures)
    sla: TenantSLA = Field(default_factory=TenantSLA)
    security: TenantSecurityConfig = Field(default_factory=TenantSecurityConfig)
    monitoring: TenantMonitoringConfig = Field(default_factory=TenantMonitoringConfig)
    billing: TenantBillingConfig = Field(default_factory=TenantBillingConfig)
    
    # Compliance et isolation
    compliance_levels: List[ComplianceLevel] = Field(default_factory=lambda: [ComplianceLevel.BASIC])
    isolation_level: IsolationLevel = Field(IsolationLevel.MODERATE)
    
    # Données administratives
    admin_email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    contact_phone: Optional[str] = Field(None, regex=r'^\+?[1-9]\d{1,14}$')
    organization: Optional[str] = Field(None, max_length=100)
    country_code: str = Field(..., regex="^[A-Z]{2}$", description="Code pays ISO")
    timezone: str = Field("UTC", description="Fuseau horaire")
    
    # Configuration technique
    custom_domain: Optional[HttpUrl] = Field(None, description="Domaine personnalisé")
    webhook_url: Optional[HttpUrl] = Field(None, description="URL webhook")
    api_endpoints: Dict[str, HttpUrl] = Field(default_factory=dict)
    
    # Métadonnées étendues
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags personnalisés")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées")
    
    class Config:
        """Configuration Pydantic."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "tenant_name": "Acme Corporation",
                "tenant_type": "enterprise",
                "admin_email": "admin@acme.com",
                "country_code": "US",
                "features": {
                    "advanced_analytics": True,
                    "custom_alerts": True,
                    "max_users": 1000,
                    "max_storage_gb": 1000
                },
                "compliance_levels": ["gdpr", "soc2"]
            }
        }
    
    @validator('updated_at', always=True)
    def set_updated_at(cls, v):
        """Met à jour automatiquement le timestamp."""
        return v or datetime.now(timezone.utc)
    
    @validator('expires_at')
    def validate_expiry(cls, v, values):
        """Valide la date d'expiration."""
        if v and 'created_at' in values:
            if v <= values['created_at']:
                raise ValueError("expires_at must be after created_at")
        return v
    
    @root_validator
    def validate_tenant_type_features(cls, values):
        """Valide la cohérence entre type de tenant et fonctionnalités."""
        tenant_type = values.get('tenant_type')
        features = values.get('features')
        
        if not tenant_type or not features:
            return values
        
        # Règles de validation par type
        if tenant_type == TenantType.TRIAL:
            if features.max_users and features.max_users > 10:
                raise ValueError("Trial tenants limited to 10 users")
            if features.max_storage_gb and features.max_storage_gb > 10:
                raise ValueError("Trial tenants limited to 10GB storage")
        
        elif tenant_type == TenantType.STANDARD:
            if features.max_users and features.max_users > 100:
                raise ValueError("Standard tenants limited to 100 users")
        
        return values
    
    @root_validator
    def validate_compliance_security(cls, values):
        """Valide la cohérence entre compliance et sécurité."""
        compliance_levels = values.get('compliance_levels', [])
        security = values.get('security')
        
        if not security:
            return values
        
        # HIPAA nécessite MFA
        if ComplianceLevel.HIPAA in compliance_levels:
            if not security.mfa_required:
                raise ValueError("HIPAA compliance requires MFA")
        
        # PCI DSS nécessite chiffrement fort
        if ComplianceLevel.PCI_DSS in compliance_levels:
            if security.encryption_level not in ["AES-256", "ChaCha20-Poly1305"]:
                raise ValueError("PCI DSS requires strong encryption")
        
        return values
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Vérifie si une fonctionnalité est activée."""
        return getattr(self.features, feature_name, False)
    
    def get_compliance_requirements(self) -> Dict[str, List[str]]:
        """Retourne les exigences de compliance."""
        requirements = {}
        for level in self.compliance_levels:
            if level == ComplianceLevel.GDPR:
                requirements[level] = [
                    "data_encryption",
                    "audit_logging",
                    "right_to_erasure",
                    "data_portability"
                ]
            elif level == ComplianceLevel.HIPAA:
                requirements[level] = [
                    "mfa_required",
                    "audit_logging",
                    "data_encryption",
                    "access_controls"
                ]
            # Ajouter d'autres niveaux...
        return requirements
    
    def validate_sla_compliance(self) -> Dict[str, bool]:
        """Valide la conformité aux SLA."""
        return {
            "uptime_target": self.sla.uptime_percentage >= 99.0,
            "response_time": self.sla.response_time_ms <= 1000,
            "backup_frequency": self.sla.backup_frequency_hours <= 24,
            "support_response": self.sla.support_response_minutes <= 240
        }


class TenantConfigUpdateSchema(BaseModel):
    """Schéma pour la mise à jour partielle de configuration tenant."""
    tenant_name: Optional[str] = Field(None, min_length=2, max_length=100)
    status: Optional[TenantStatus] = None
    features: Optional[TenantFeatures] = None
    sla: Optional[TenantSLA] = None
    security: Optional[TenantSecurityConfig] = None
    monitoring: Optional[TenantMonitoringConfig] = None
    billing: Optional[TenantBillingConfig] = None
    compliance_levels: Optional[List[ComplianceLevel]] = None
    isolation_level: Optional[IsolationLevel] = None
    admin_email: Optional[str] = Field(None, regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    contact_phone: Optional[str] = Field(None, regex=r'^\+?[1-9]\d{1,14}$')
    organization: Optional[str] = Field(None, max_length=100)
    timezone: Optional[str] = None
    custom_domain: Optional[HttpUrl] = None
    webhook_url: Optional[HttpUrl] = None
    tags: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "forbid"
