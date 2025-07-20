"""
Data Models - Modèles de Données Analytics
==========================================

Ce module définit tous les modèles de données utilisés dans le système
d'analytics, incluant les métriques, événements, alertes, tableaux de bord,
tenants, utilisateurs et sessions.

Classes principales:
- Metric: Modèle de métrique
- Event: Modèle d'événement
- Alert: Modèle d'alerte
- Dashboard: Modèle de tableau de bord
- Tenant: Modèle de tenant
- User: Modèle d'utilisateur
- Session: Modèle de session
"""

import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal

from pydantic import BaseModel, Field, validator, root_validator
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship


# Base SQLAlchemy
Base = declarative_base()


class MetricType(str, Enum):
    """Types de métriques."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class EventType(str, Enum):
    """Types d'événements."""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    BUSINESS_EVENT = "business_event"
    ERROR_EVENT = "error_event"
    SECURITY_EVENT = "security_event"


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """États des alertes."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class DashboardType(str, Enum):
    """Types de tableaux de bord."""
    REALTIME = "realtime"
    ANALYTICAL = "analytical"
    EXECUTIVE = "executive"
    OPERATIONAL = "operational"
    CUSTOM = "custom"


class UserRole(str, Enum):
    """Rôles utilisateur."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    TENANT_ADMIN = "tenant_admin"
    TENANT_USER = "tenant_user"


class SessionStatus(str, Enum):
    """États de session."""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"


# ==================== Modèles Pydantic ====================

class BaseAnalyticsModel(BaseModel):
    """Modèle de base pour tous les modèles analytics."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: float
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return self.dict()
    
    def to_json(self) -> str:
        """Convertit en JSON."""
        return self.json()


class Metric(BaseAnalyticsModel):
    """Modèle de métrique."""
    
    name: str = Field(..., description="Nom de la métrique")
    value: Union[int, float, Decimal] = Field(..., description="Valeur de la métrique")
    metric_type: MetricType = Field(default=MetricType.GAUGE, description="Type de métrique")
    unit: Optional[str] = Field(None, description="Unité de mesure")
    tenant_id: str = Field(..., description="ID du tenant")
    user_id: Optional[str] = Field(None, description="ID de l'utilisateur")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags de la métrique")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Horodatage")
    ttl: Optional[int] = Field(None, description="TTL en secondes")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Le nom de la métrique doit faire au moins 2 caractères")
        return v.lower().replace(' ', '_')
    
    @validator('value')
    def validate_value(cls, v):
        if v is None:
            raise ValueError("La valeur de la métrique ne peut pas être nulle")
        return float(v)
    
    @validator('tags')
    def validate_tags(cls, v):
        if not isinstance(v, dict):
            return {}
        # Nettoyer les tags
        return {k.lower(): str(val) for k, val in v.items() if k and val}
    
    @property
    def metric_key(self) -> str:
        """Clé unique de la métrique."""
        tags_str = ",".join(f"{k}={v}" for k, v in sorted(self.tags.items()))
        return f"{self.tenant_id}:{self.name}:{tags_str}"
    
    def add_tag(self, key: str, value: str):
        """Ajoute un tag."""
        self.tags[key.lower()] = str(value)
    
    def remove_tag(self, key: str):
        """Supprime un tag."""
        self.tags.pop(key.lower(), None)
    
    def has_tag(self, key: str, value: Optional[str] = None) -> bool:
        """Vérifie si un tag existe."""
        key = key.lower()
        if key not in self.tags:
            return False
        if value is None:
            return True
        return self.tags[key] == str(value)


class Event(BaseAnalyticsModel):
    """Modèle d'événement."""
    
    name: str = Field(..., description="Nom de l'événement")
    event_type: EventType = Field(..., description="Type d'événement")
    tenant_id: str = Field(..., description="ID du tenant")
    user_id: Optional[str] = Field(None, description="ID de l'utilisateur")
    session_id: Optional[str] = Field(None, description="ID de session")
    source: str = Field(..., description="Source de l'événement")
    data: Dict[str, Any] = Field(default_factory=dict, description="Données de l'événement")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contexte de l'événement")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags de l'événement")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Horodatage")
    processed: bool = Field(default=False, description="Événement traité")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Le nom de l'événement doit faire au moins 2 caractères")
        return v.lower().replace(' ', '_')
    
    @validator('source')
    def validate_source(cls, v):
        if not v:
            raise ValueError("La source de l'événement est obligatoire")
        return v
    
    @property
    def event_key(self) -> str:
        """Clé unique de l'événement."""
        return f"{self.tenant_id}:{self.event_type}:{self.name}:{self.timestamp.isoformat()}"
    
    def add_context(self, key: str, value: Any):
        """Ajoute du contexte."""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Récupère du contexte."""
        return self.context.get(key, default)


class Alert(BaseAnalyticsModel):
    """Modèle d'alerte."""
    
    name: str = Field(..., description="Nom de l'alerte")
    title: str = Field(..., description="Titre de l'alerte")
    message: str = Field(..., description="Message de l'alerte")
    severity: AlertSeverity = Field(..., description="Sévérité de l'alerte")
    status: AlertStatus = Field(default=AlertStatus.ACTIVE, description="État de l'alerte")
    tenant_id: str = Field(..., description="ID du tenant")
    rule_id: Optional[str] = Field(None, description="ID de la règle")
    metric_name: Optional[str] = Field(None, description="Nom de la métrique")
    threshold_value: Optional[float] = Field(None, description="Valeur seuil")
    current_value: Optional[float] = Field(None, description="Valeur actuelle")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags de l'alerte")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées")
    channels: List[str] = Field(default_factory=list, description="Canaux de notification")
    acknowledged_by: Optional[str] = Field(None, description="Acquittée par")
    acknowledged_at: Optional[datetime] = Field(None, description="Acquittée à")
    resolved_by: Optional[str] = Field(None, description="Résolue par")
    resolved_at: Optional[datetime] = Field(None, description="Résolue à")
    expires_at: Optional[datetime] = Field(None, description="Expire à")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Le nom de l'alerte doit faire au moins 2 caractères")
        return v
    
    @validator('message')
    def validate_message(cls, v):
        if not v:
            raise ValueError("Le message de l'alerte est obligatoire")
        return v
    
    @validator('channels')
    def validate_channels(cls, v):
        valid_channels = ['email', 'slack', 'webhook', 'sms', 'push']
        return [ch for ch in v if ch in valid_channels]
    
    def acknowledge(self, user_id: str):
        """Acquitte l'alerte."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user_id
        self.acknowledged_at = datetime.utcnow()
    
    def resolve(self, user_id: str):
        """Résout l'alerte."""
        self.status = AlertStatus.RESOLVED
        self.resolved_by = user_id
        self.resolved_at = datetime.utcnow()
    
    def suppress(self):
        """Supprime l'alerte."""
        self.status = AlertStatus.SUPPRESSED
    
    @property
    def is_active(self) -> bool:
        """Vérifie si l'alerte est active."""
        return self.status == AlertStatus.ACTIVE
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Durée de l'alerte."""
        if self.resolved_at:
            return self.resolved_at - self.created_at
        return datetime.utcnow() - self.created_at


class Dashboard(BaseAnalyticsModel):
    """Modèle de tableau de bord."""
    
    name: str = Field(..., description="Nom du tableau de bord")
    title: str = Field(..., description="Titre du tableau de bord")
    description: Optional[str] = Field(None, description="Description")
    dashboard_type: DashboardType = Field(..., description="Type de tableau de bord")
    tenant_id: str = Field(..., description="ID du tenant")
    owner_id: str = Field(..., description="ID du propriétaire")
    layout: Dict[str, Any] = Field(default_factory=dict, description="Configuration de layout")
    widgets: List[Dict[str, Any]] = Field(default_factory=list, description="Widgets")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filtres")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Paramètres")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags")
    is_public: bool = Field(default=False, description="Tableau de bord public")
    is_template: bool = Field(default=False, description="Template")
    shared_with: List[str] = Field(default_factory=list, description="Partagé avec")
    refresh_interval: int = Field(default=300, description="Intervalle de rafraîchissement (secondes)")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Le nom du tableau de bord doit faire au moins 2 caractères")
        return v
    
    @validator('refresh_interval')
    def validate_refresh_interval(cls, v):
        if v < 30:
            return 30  # Minimum 30 secondes
        return v
    
    def add_widget(self, widget_config: Dict[str, Any]):
        """Ajoute un widget."""
        widget_config['id'] = str(uuid.uuid4())
        widget_config['created_at'] = datetime.utcnow().isoformat()
        self.widgets.append(widget_config)
    
    def remove_widget(self, widget_id: str):
        """Supprime un widget."""
        self.widgets = [w for w in self.widgets if w.get('id') != widget_id]
    
    def get_widget(self, widget_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un widget."""
        for widget in self.widgets:
            if widget.get('id') == widget_id:
                return widget
        return None
    
    def share_with_user(self, user_id: str):
        """Partage avec un utilisateur."""
        if user_id not in self.shared_with:
            self.shared_with.append(user_id)
    
    def unshare_with_user(self, user_id: str):
        """Annule le partage avec un utilisateur."""
        if user_id in self.shared_with:
            self.shared_with.remove(user_id)


class Tenant(BaseAnalyticsModel):
    """Modèle de tenant."""
    
    name: str = Field(..., description="Nom du tenant")
    slug: str = Field(..., description="Slug unique du tenant")
    domain: Optional[str] = Field(None, description="Domaine du tenant")
    plan: str = Field(default="basic", description="Plan tarifaire")
    status: str = Field(default="active", description="Statut du tenant")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Paramètres du tenant")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées")
    features: List[str] = Field(default_factory=list, description="Fonctionnalités activées")
    limits: Dict[str, int] = Field(default_factory=dict, description="Limites du tenant")
    owner_id: str = Field(..., description="ID du propriétaire")
    is_active: bool = Field(default=True, description="Tenant actif")
    trial_expires_at: Optional[datetime] = Field(None, description="Fin d'essai")
    
    @validator('slug')
    def validate_slug(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Le slug doit faire au moins 2 caractères")
        # Nettoyer le slug
        import re
        return re.sub(r'[^a-z0-9-]', '', v.lower())
    
    @validator('plan')
    def validate_plan(cls, v):
        valid_plans = ['basic', 'pro', 'enterprise', 'trial']
        if v not in valid_plans:
            return 'basic'
        return v
    
    def add_feature(self, feature: str):
        """Ajoute une fonctionnalité."""
        if feature not in self.features:
            self.features.append(feature)
    
    def remove_feature(self, feature: str):
        """Supprime une fonctionnalité."""
        if feature in self.features:
            self.features.remove(feature)
    
    def has_feature(self, feature: str) -> bool:
        """Vérifie si une fonctionnalité est activée."""
        return feature in self.features
    
    def set_limit(self, limit_type: str, value: int):
        """Définit une limite."""
        self.limits[limit_type] = value
    
    def get_limit(self, limit_type: str, default: int = 0) -> int:
        """Récupère une limite."""
        return self.limits.get(limit_type, default)
    
    @property
    def is_trial(self) -> bool:
        """Vérifie si le tenant est en période d'essai."""
        return self.plan == 'trial' and self.trial_expires_at and self.trial_expires_at > datetime.utcnow()


class User(BaseAnalyticsModel):
    """Modèle d'utilisateur."""
    
    username: str = Field(..., description="Nom d'utilisateur")
    email: str = Field(..., description="Email")
    first_name: Optional[str] = Field(None, description="Prénom")
    last_name: Optional[str] = Field(None, description="Nom de famille")
    role: UserRole = Field(default=UserRole.VIEWER, description="Rôle")
    tenant_id: str = Field(..., description="ID du tenant")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Préférences")
    permissions: List[str] = Field(default_factory=list, description="Permissions")
    is_active: bool = Field(default=True, description="Utilisateur actif")
    last_login: Optional[datetime] = Field(None, description="Dernière connexion")
    login_count: int = Field(default=0, description="Nombre de connexions")
    
    @validator('email')
    def validate_email(cls, v):
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError("Format d'email invalide")
        return v.lower()
    
    @validator('username')
    def validate_username(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Le nom d'utilisateur doit faire au moins 2 caractères")
        return v.lower()
    
    def add_permission(self, permission: str):
        """Ajoute une permission."""
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def remove_permission(self, permission: str):
        """Supprime une permission."""
        if permission in self.permissions:
            self.permissions.remove(permission)
    
    def has_permission(self, permission: str) -> bool:
        """Vérifie si l'utilisateur a une permission."""
        return permission in self.permissions
    
    def record_login(self):
        """Enregistre une connexion."""
        self.last_login = datetime.utcnow()
        self.login_count += 1
    
    @property
    def full_name(self) -> str:
        """Nom complet."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or self.last_name or self.username


class Session(BaseAnalyticsModel):
    """Modèle de session."""
    
    session_token: str = Field(..., description="Token de session")
    user_id: str = Field(..., description="ID de l'utilisateur")
    tenant_id: str = Field(..., description="ID du tenant")
    ip_address: Optional[str] = Field(None, description="Adresse IP")
    user_agent: Optional[str] = Field(None, description="User Agent")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Statut de la session")
    expires_at: datetime = Field(..., description="Expiration")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Dernière activité")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées de session")
    
    @validator('session_token')
    def validate_session_token(cls, v):
        if not v or len(v) < 32:
            raise ValueError("Le token de session doit faire au moins 32 caractères")
        return v
    
    @validator('expires_at')
    def validate_expires_at(cls, v):
        if v <= datetime.utcnow():
            raise ValueError("La date d'expiration doit être dans le futur")
        return v
    
    def extend_session(self, hours: int = 2):
        """Prolonge la session."""
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.last_activity = datetime.utcnow()
    
    def terminate(self):
        """Termine la session."""
        self.status = SessionStatus.TERMINATED
        self.expires_at = datetime.utcnow()
    
    def suspend(self):
        """Suspend la session."""
        self.status = SessionStatus.SUSPENDED
    
    def reactivate(self):
        """Réactive la session."""
        if self.status == SessionStatus.SUSPENDED:
            self.status = SessionStatus.ACTIVE
            self.last_activity = datetime.utcnow()
    
    @property
    def is_expired(self) -> bool:
        """Vérifie si la session est expirée."""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_active(self) -> bool:
        """Vérifie si la session est active."""
        return self.status == SessionStatus.ACTIVE and not self.is_expired
    
    @property
    def time_remaining(self) -> timedelta:
        """Temps restant avant expiration."""
        if self.is_expired:
            return timedelta(0)
        return self.expires_at - datetime.utcnow()


# ==================== Modèles SQLAlchemy ====================

class MetricDB(Base):
    """Modèle SQLAlchemy pour les métriques."""
    __tablename__ = "analytics_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)
    unit = Column(String(50))
    tenant_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), index=True)
    tags = Column(JSON)
    metadata = Column(JSON)
    timestamp = Column(DateTime, nullable=False, index=True)
    ttl = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class EventDB(Base):
    """Modèle SQLAlchemy pour les événements."""
    __tablename__ = "analytics_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)
    tenant_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), index=True)
    session_id = Column(String(255), index=True)
    source = Column(String(255), nullable=False)
    data = Column(JSON)
    context = Column(JSON)
    tags = Column(JSON)
    timestamp = Column(DateTime, nullable=False, index=True)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AlertDB(Base):
    """Modèle SQLAlchemy pour les alertes."""
    __tablename__ = "analytics_alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    title = Column(String(500), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    tenant_id = Column(String(255), nullable=False, index=True)
    rule_id = Column(String(255))
    metric_name = Column(String(255))
    threshold_value = Column(Float)
    current_value = Column(Float)
    tags = Column(JSON)
    metadata = Column(JSON)
    channels = Column(JSON)
    acknowledged_by = Column(String(255))
    acknowledged_at = Column(DateTime)
    resolved_by = Column(String(255))
    resolved_at = Column(DateTime)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)


class DashboardDB(Base):
    """Modèle SQLAlchemy pour les tableaux de bord."""
    __tablename__ = "analytics_dashboards"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    dashboard_type = Column(String(50), nullable=False)
    tenant_id = Column(String(255), nullable=False, index=True)
    owner_id = Column(String(255), nullable=False)
    layout = Column(JSON)
    widgets = Column(JSON)
    filters = Column(JSON)
    settings = Column(JSON)
    tags = Column(JSON)
    is_public = Column(Boolean, default=False)
    is_template = Column(Boolean, default=False)
    shared_with = Column(JSON)
    refresh_interval = Column(Integer, default=300)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)


# Fonctions utilitaires
def create_metric(**kwargs) -> Metric:
    """Crée une nouvelle métrique."""
    return Metric(**kwargs)


def create_event(**kwargs) -> Event:
    """Crée un nouvel événement."""
    return Event(**kwargs)


def create_alert(**kwargs) -> Alert:
    """Crée une nouvelle alerte."""
    return Alert(**kwargs)


def create_dashboard(**kwargs) -> Dashboard:
    """Crée un nouveau tableau de bord."""
    return Dashboard(**kwargs)


def create_tenant(**kwargs) -> Tenant:
    """Crée un nouveau tenant."""
    return Tenant(**kwargs)


def create_user(**kwargs) -> User:
    """Crée un nouvel utilisateur."""
    return User(**kwargs)


def create_session(**kwargs) -> Session:
    """Crée une nouvelle session."""
    return Session(**kwargs)
