"""
Schémas de configuration pour tenant - Module Python.

Ce module fournit les classes de validation et de sérialisation pour
les configurations tenant basées sur les schémas JSON.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class EnvironmentType(str, Enum):
    """Types d'environnement supportés."""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    TEST = "test"


class PlanType(str, Enum):
    """Types de plans de facturation."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class AuthProvider(str, Enum):
    """Fournisseurs d'authentification supportés."""
    OAUTH2 = "oauth2"
    SAML = "saml"
    LDAP = "ldap"
    JWT = "jwt"


class TenantOwner(BaseModel):
    """Propriétaire du tenant."""
    user_id: str = Field(..., description="ID utilisateur")
    email: str = Field(..., description="Email du propriétaire")
    name: str = Field(..., description="Nom du propriétaire")


class TenantMetadata(BaseModel):
    """Métadonnées du tenant."""
    name: str = Field(..., min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    owner: TenantOwner
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class DatabaseConfig(BaseModel):
    """Configuration de base de données."""
    host: str
    port: int = Field(..., ge=1, le=65535)
    name: str
    schema: str
    ssl_mode: str = Field("require", regex="^(require|prefer|allow|disable)$")
    connection_pool: Optional[Dict[str, int]] = None


class CacheConfig(BaseModel):
    """Configuration du cache Redis."""
    redis: Optional[Dict[str, Any]] = None


class StorageConfig(BaseModel):
    """Configuration du stockage."""
    type: str = Field(..., regex="^(s3|gcs|azure|local)$")
    bucket: str
    prefix: Optional[str] = None
    encryption: bool = True


class MonitoringConfig(BaseModel):
    """Configuration du monitoring."""
    enabled: bool = True
    metrics: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None


class EnvironmentConfig(BaseModel):
    """Configuration d'un environnement."""
    enabled: bool = True
    database: DatabaseConfig
    cache: Optional[CacheConfig] = None
    storage: Optional[StorageConfig] = None
    monitoring: Optional[MonitoringConfig] = None


class AIProcessingFeature(BaseModel):
    """Configuration des fonctionnalités IA."""
    enabled: bool = True
    models: List[str] = Field(default_factory=list)
    rate_limits: Optional[Dict[str, int]] = None


class CollaborationFeature(BaseModel):
    """Configuration des fonctionnalités de collaboration."""
    enabled: bool = True
    max_users: int = Field(10, ge=1)
    real_time: bool = True


class SpotifyIntegration(BaseModel):
    """Configuration de l'intégration Spotify."""
    enabled: bool = True
    api_version: str = "v1"
    scopes: List[str] = Field(default_factory=list)


class TenantFeatures(BaseModel):
    """Fonctionnalités du tenant."""
    ai_processing: Optional[AIProcessingFeature] = None
    collaboration: Optional[CollaborationFeature] = None
    spotify_integration: Optional[SpotifyIntegration] = None


class AuthenticationConfig(BaseModel):
    """Configuration d'authentification."""
    provider: AuthProvider = AuthProvider.JWT
    mfa_required: bool = False
    session_timeout: int = Field(3600, ge=300)


class EncryptionConfig(BaseModel):
    """Configuration du chiffrement."""
    at_rest: bool = True
    in_transit: bool = True
    algorithm: str = Field("AES-256", regex="^(AES-256|ChaCha20-Poly1305)$")


class DataPrivacyConfig(BaseModel):
    """Configuration de confidentialité des données."""
    gdpr_compliant: bool = True
    data_retention_days: int = Field(365, ge=1)
    anonymization: bool = True


class SecurityConfig(BaseModel):
    """Configuration de sécurité."""
    authentication: AuthenticationConfig
    encryption: EncryptionConfig
    data_privacy: DataPrivacyConfig


class UsageLimits(BaseModel):
    """Limites d'utilisation."""
    api_calls: Optional[int] = None
    storage_gb: Optional[float] = None
    ai_minutes: Optional[int] = None


class BillingConfig(BaseModel):
    """Configuration de facturation."""
    plan: PlanType = PlanType.FREE
    billing_cycle: str = Field("monthly", regex="^(monthly|yearly)$")
    usage_limits: Optional[UsageLimits] = None


class TenantConfigSchema(BaseModel):
    """Schéma complet de configuration tenant."""
    tenant_id: str = Field(..., regex="^[a-zA-Z0-9-_]{3,50}$")
    metadata: TenantMetadata
    environments: Dict[EnvironmentType, EnvironmentConfig]
    features: TenantFeatures
    security: SecurityConfig
    billing: BillingConfig

    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        """Valide l'ID du tenant."""
        if not v or len(v) < 3:
            raise ValueError("L'ID du tenant doit contenir au moins 3 caractères")
        return v.lower()

    @validator('environments')
    def validate_environments(cls, v):
        """Valide qu'au moins un environnement est configuré."""
        if not v:
            raise ValueError("Au moins un environnement doit être configuré")
        return v

    class Config:
        """Configuration Pydantic."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"


class TenantMetadataSchema(BaseModel):
    """Schéma simplifié pour les métadonnées tenant."""
    tenant_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    owner_email: str
    plan: PlanType
    environments: List[EnvironmentType]
    features_enabled: List[str]

    class Config:
        """Configuration Pydantic."""
        use_enum_values = True
