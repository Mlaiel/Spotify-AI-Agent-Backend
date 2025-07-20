"""
🏢 Tenant Model - Modèle Principal Multi-Tenant
==============================================

Modèle de données central pour la gestion des tenants.
Définit la structure principale d'un tenant et ses configurations.

Author: DBA & Data Engineer - Fahed Mlaiel
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

from sqlalchemy import Column, String, DateTime, Boolean, JSON, Integer, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, validator

Base = declarative_base()


class TenantTier(str, Enum):
    """Niveaux de tenant"""
    FREE = "free"
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class TenantStatus(str, Enum):
    """États du tenant"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    TERMINATED = "terminated"


class IsolationLevel(str, Enum):
    """Niveaux d'isolation"""
    SHARED = "shared"                # Base partagée
    SCHEMA = "schema"                # Schéma dédié
    DATABASE = "database"            # Base dédiée
    INSTANCE = "instance"            # Instance dédiée


class Tenant(Base):
    """
    Modèle principal de tenant.
    
    Attributes:
        id: Identifiant unique du tenant
        name: Nom du tenant
        domain: Domaine du tenant
        tier: Niveau de service
        status: État du tenant
        isolation_level: Niveau d'isolation des données
        settings: Configuration JSON du tenant
        metadata: Métadonnées additionnelles
        created_at: Date de création
        updated_at: Date de dernière modification
        expires_at: Date d'expiration (optionnelle)
    """
    
    __tablename__ = "tenants"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    domain = Column(String(255), unique=True, nullable=False)
    tier = Column(String(50), nullable=False, default=TenantTier.FREE)
    status = Column(String(50), nullable=False, default=TenantStatus.PENDING)
    isolation_level = Column(String(50), nullable=False, default=IsolationLevel.SHARED)
    
    # Configuration et métadonnées
    settings = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Relations
    users = relationship("TenantUser", back_populates="tenant", cascade="all, delete-orphan")
    subscriptions = relationship("Subscription", back_populates="tenant", cascade="all, delete-orphan")
    features = relationship("TenantFeature", back_populates="tenant", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Tenant(id='{self.id}', name='{self.name}', domain='{self.domain}')>"
    
    @property
    def is_active(self) -> bool:
        """Vérifier si le tenant est actif"""
        return self.status == TenantStatus.ACTIVE
    
    @property
    def is_expired(self) -> bool:
        """Vérifier si le tenant est expiré"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Récupérer une configuration"""
        return self.settings.get(key, default)
    
    def set_setting(self, key: str, value: Any):
        """Définir une configuration"""
        if not self.settings:
            self.settings = {}
        self.settings[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Récupérer une métadonnée"""
        return self.metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any):
        """Définir une métadonnée"""
        if not self.metadata:
            self.metadata = {}
        self.metadata[key] = value


class TenantSettings(Base):
    """
    Configuration détaillée du tenant.
    
    Permet de stocker des configurations complexes
    avec versioning et validation.
    """
    
    __tablename__ = "tenant_settings"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    category = Column(String(100), nullable=False)  # e.g., "security", "billing", "features"
    key = Column(String(255), nullable=False)
    value = Column(JSON, nullable=False)
    value_type = Column(String(50), nullable=False)  # "string", "number", "boolean", "object", "array"
    description = Column(Text)
    is_encrypted = Column(Boolean, default=False)
    version = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<TenantSettings(tenant_id='{self.tenant_id}', key='{self.key}')>"


class TenantFeature(Base):
    """
    Fonctionnalités activées pour un tenant.
    
    Gère les features flags et les limitations
    par niveau de service.
    """
    
    __tablename__ = "tenant_features"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    feature_name = Column(String(255), nullable=False)
    enabled = Column(Boolean, default=True)
    configuration = Column(JSON, default=dict)
    limits = Column(JSON, default=dict)  # e.g., {"max_users": 100, "max_storage": "10GB"}
    
    enabled_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Relations
    tenant = relationship("Tenant", back_populates="features")
    
    def __repr__(self):
        return f"<TenantFeature(tenant_id='{self.tenant_id}', feature='{self.feature_name}')>"
    
    @property
    def is_active(self) -> bool:
        """Vérifier si la fonctionnalité est active"""
        if not self.enabled:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def get_limit(self, limit_name: str, default: Any = None) -> Any:
        """Récupérer une limite"""
        return self.limits.get(limit_name, default)
    
    def set_limit(self, limit_name: str, value: Any):
        """Définir une limite"""
        if not self.limits:
            self.limits = {}
        self.limits[limit_name] = value


# Modèles Pydantic pour l'API

class TenantCreate(BaseModel):
    """Modèle de création de tenant"""
    name: str
    domain: str
    tier: TenantTier = TenantTier.FREE
    isolation_level: IsolationLevel = IsolationLevel.SHARED
    settings: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    expires_at: Optional[datetime] = None
    
    @validator("domain")
    def validate_domain(cls, v):
        if not v or len(v) < 3:
            raise ValueError("Le domaine doit contenir au moins 3 caractères")
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Le domaine ne peut contenir que des lettres, chiffres, tirets et underscores")
        return v.lower()
    
    @validator("name")
    def validate_name(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Le nom doit contenir au moins 2 caractères")
        return v


class TenantUpdate(BaseModel):
    """Modèle de mise à jour de tenant"""
    name: Optional[str] = None
    tier: Optional[TenantTier] = None
    status: Optional[TenantStatus] = None
    isolation_level: Optional[IsolationLevel] = None
    settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None


class TenantResponse(BaseModel):
    """Modèle de réponse tenant"""
    id: str
    name: str
    domain: str
    tier: TenantTier
    status: TenantStatus
    isolation_level: IsolationLevel
    settings: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    is_expired: bool
    
    class Config:
        from_attributes = True


class TenantFeatureCreate(BaseModel):
    """Modèle de création de fonctionnalité"""
    feature_name: str
    enabled: bool = True
    configuration: Dict[str, Any] = {}
    limits: Dict[str, Any] = {}
    expires_at: Optional[datetime] = None


class TenantFeatureResponse(BaseModel):
    """Modèle de réponse fonctionnalité"""
    id: str
    tenant_id: str
    feature_name: str
    enabled: bool
    configuration: Dict[str, Any]
    limits: Dict[str, Any]
    enabled_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    
    class Config:
        from_attributes = True


class TenantSettingsCreate(BaseModel):
    """Modèle de création de configuration"""
    category: str
    key: str
    value: Any
    value_type: str
    description: Optional[str] = None
    is_encrypted: bool = False


class TenantSettingsResponse(BaseModel):
    """Modèle de réponse configuration"""
    id: str
    tenant_id: str
    category: str
    key: str
    value: Any
    value_type: str
    description: Optional[str]
    is_encrypted: bool
    version: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Configuration par défaut des tenants
DEFAULT_TENANT_SETTINGS = {
    "security": {
        "password_policy": {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_symbols": False
        },
        "session_timeout": 3600,  # 1 heure
        "max_login_attempts": 5,
        "lockout_duration": 300   # 5 minutes
    },
    "billing": {
        "currency": "EUR",
        "billing_cycle": "monthly",
        "auto_renew": True,
        "payment_grace_period": 7  # jours
    },
    "features": {
        "max_users": 10,
        "max_storage": "1GB",
        "max_api_calls": 1000,
        "analytics_enabled": True,
        "backup_enabled": False
    },
    "notifications": {
        "email_enabled": True,
        "sms_enabled": False,
        "webhook_enabled": False
    },
    "data": {
        "retention_days": 365,
        "backup_frequency": "weekly",
        "compression_enabled": True,
        "encryption_enabled": True
    }
}

# Limites par niveau de service
TIER_LIMITS = {
    TenantTier.FREE: {
        "max_users": 5,
        "max_storage": "500MB",
        "max_api_calls": 500,
        "features": ["basic_analytics", "email_support"]
    },
    TenantTier.BASIC: {
        "max_users": 25,
        "max_storage": "5GB", 
        "max_api_calls": 5000,
        "features": ["analytics", "email_support", "basic_backup"]
    },
    TenantTier.STANDARD: {
        "max_users": 100,
        "max_storage": "25GB",
        "max_api_calls": 25000,
        "features": ["advanced_analytics", "priority_support", "automated_backup", "sso"]
    },
    TenantTier.PREMIUM: {
        "max_users": 500,
        "max_storage": "100GB",
        "max_api_calls": 100000,
        "features": ["premium_analytics", "24x7_support", "real_time_backup", "sso", "api_access"]
    },
    TenantTier.ENTERPRISE: {
        "max_users": -1,  # Illimité
        "max_storage": "unlimited",
        "max_api_calls": -1,  # Illimité
        "features": ["enterprise_analytics", "dedicated_support", "continuous_backup", "sso", "api_access", "custom_integrations"]
    }
}
