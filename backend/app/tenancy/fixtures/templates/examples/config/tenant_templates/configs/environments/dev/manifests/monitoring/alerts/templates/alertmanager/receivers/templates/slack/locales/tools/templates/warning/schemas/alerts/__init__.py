"""
Schémas d'alerte avancés - Spotify AI Agent
Module principal pour la gestion des alertes multi-tenant avec ML et analytics
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


# Énumérations de base (fallback si modules spécialisés non disponibles)
class AlertLevel(str, Enum):
    """Niveaux d'alerte"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """États d'alerte"""
    PENDING = "pending"
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class WarningCategory(str, Enum):
    """Catégories d'avertissement"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    AVAILABILITY = "availability"
    CAPACITY = "capacity"
    COMPLIANCE = "compliance"
    QUALITY = "quality"


class Priority(str, Enum):
    """Niveaux de priorité"""
    P0 = "p0"  # Critique
    P1 = "p1"  # Élevé
    P2 = "p2"  # Moyen
    P3 = "p3"  # Bas
    P4 = "p4"  # Info
    P5 = "p5"  # Minimal


class Environment(str, Enum):
    """Environnements"""
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TEST = "test"
    LOCAL = "local"


class NotificationChannel(str, Enum):
    """Canaux de notification"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"
    PUSH = "push"
    CALL = "call"


class EscalationLevel(str, Enum):
    """Niveaux d'escalade"""
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class CorrelationMethod(str, Enum):
    """Méthodes de corrélation"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CAUSAL = "causal"
    PATTERN = "pattern"
    ML_CLUSTERING = "ml_clustering"
    GRAPH_ANALYSIS = "graph_analysis"


class WorkflowStatus(str, Enum):
    """États de workflow"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class IncidentStatus(str, Enum):
    """États d'incident"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ModelFramework(str, Enum):
    """Frameworks ML"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class SecurityLevel(str, Enum):
    """Niveaux de sécurité"""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


# Mixins de base
class BaseSchema(BaseModel):
    """Schéma de base avec configuration commune"""
    id: UUID = Field(default_factory=uuid4, description="Identifiant unique")
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid',
        frozen=False
    )


class TimestampMixin(BaseModel):
    """Mixin pour les timestamps"""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deleted_at: Optional[datetime] = Field(None)

    def mark_updated(self):
        """Met à jour le timestamp de modification"""
        self.updated_at = datetime.now(timezone.utc)

    def mark_deleted(self):
        """Marque comme supprimé (soft delete)"""
        self.deleted_at = datetime.now(timezone.utc)

    @property
    def is_deleted(self) -> bool:
        """Indique si l'objet est supprimé"""
        return self.deleted_at is not None


class TenantMixin(BaseModel):
    """Mixin pour la multi-tenancy"""
    tenant_id: Optional[UUID] = Field(None, description="Identifiant du tenant")
    organization_id: Optional[UUID] = Field(None, description="Identifiant de l'organisation")


class MetadataMixin(BaseModel):
    """Mixin pour les métadonnées"""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: Set[str] = Field(default_factory=set)
    labels: Dict[str, str] = Field(default_factory=dict)

    def add_tag(self, tag: str):
        """Ajoute un tag"""
        self.tags.add(tag)

    def remove_tag(self, tag: str):
        """Supprime un tag"""
        self.tags.discard(tag)

    def set_metadata(self, key: str, value: Any):
        """Définit une métadonnée"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Récupère une métadonnée"""
        return self.metadata.get(key, default)


# Imports conditionnels des modules spécialisés
try:
    from .rules import *
except ImportError:
    pass

try:
    from .notifications import *
except ImportError:
    pass

try:
    from .escalation import *
except ImportError:
    pass

try:
    from .correlation import *
except ImportError:
    pass

try:
    from .analytics import *
except ImportError:
    pass

try:
    from .templates import *
except ImportError:
    pass

try:
    from .workflows import *
except ImportError:
    pass

try:
    from .incidents import *
except ImportError:
    pass

try:
    from .compliance import *
except ImportError:
    pass

try:
    from .ml_models import *
except ImportError:
    pass

try:
    from .webhooks import *
except ImportError:
    pass

try:
    from .validations import *
except ImportError:
    pass

try:
    from .utils import *
except ImportError:
    pass


__version__ = "1.0.0"
__author__ = "Fahed Mlaiel"
__description__ = "Système complet de gestion d'alertes avancé avec ML et analytics"


__all__ = [
    # Métadonnées du module
    '__version__', '__author__', '__description__',
    
    # Énumérations de base
    'AlertLevel', 'AlertStatus', 'WarningCategory', 'Priority', 'Environment',
    'NotificationChannel', 'EscalationLevel', 'CorrelationMethod', 'WorkflowStatus',
    'IncidentStatus', 'ModelFramework', 'SecurityLevel',
    
    # Classes de base
    'BaseSchema', 'TimestampMixin', 'TenantMixin', 'MetadataMixin',
    
    # Toutes les exportations des modules spécialisés seront automatiquement
    # ajoutées via les imports avec '*' ci-dessus
]

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Literal, Annotated
from uuid import UUID, uuid4
from decimal import Decimal
from enum import Enum
import json
import re
from pathlib import Path

from pydantic import (
    BaseModel, Field, validator, root_validator, computed_field,
    EmailStr, HttpUrl, IPvAnyAddress, StrictStr, ConfigDict
)

# Import base schemas and enums
try:
    from ..base import BaseSchema, TimestampMixin, TenantMixin, MetadataMixin
    from ..base.enums import (
        AlertLevel, AlertStatus, WarningCategory, NotificationChannel, 
        Priority, Environment, ProcessingStatus
    )
except ImportError:
    # Fallback definitions for standalone usage
    class BaseSchema(BaseModel):
        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_assignment=True,
            use_enum_values=True,
            extra='forbid'
        )
    
    class TimestampMixin(BaseModel):
        created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
        updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class TenantMixin(BaseModel):
        tenant_id: UUID = Field(...)
    
    class MetadataMixin(BaseModel):
        metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class AlertLevel(str, Enum):
        DEBUG = "debug"
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"
        FATAL = "fatal"
    
    class AlertStatus(str, Enum):
        PENDING = "pending"
        ACTIVE = "active"
        ACKNOWLEDGED = "acknowledged"
        RESOLVED = "resolved"
        SUPPRESSED = "suppressed"
        EXPIRED = "expired"
    
    class WarningCategory(str, Enum):
        SYSTEM = "system"
        APPLICATION = "application"
        SECURITY = "security"
        PERFORMANCE = "performance"
        BUSINESS = "business"
        INFRASTRUCTURE = "infrastructure"
    
    class NotificationChannel(str, Enum):
        EMAIL = "email"
        SLACK = "slack"
        SMS = "sms"
        WEBHOOK = "webhook"
        PAGERDUTY = "pagerduty"
        OPSGENIE = "opsgenie"
    
    class Priority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        URGENT = "urgent"
        CRITICAL = "critical"
    
    class Environment(str, Enum):
        DEVELOPMENT = "development"
        STAGING = "staging"
        PRODUCTION = "production"
        TEST = "test"
    
    class ProcessingStatus(str, Enum):
        QUEUED = "queued"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"
        RETRYING = "retrying"


class AlertRule(BaseSchema):
    """Règle de déclenchement d'alerte"""
    name: StrictStr = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    category: WarningCategory = Field(...)
    
    # Conditions de déclenchement
    trigger_condition: str = Field(..., description="Condition de déclenchement")
    threshold_value: Optional[float] = Field(None, description="Valeur seuil")
    threshold_operator: str = Field(">=", regex=r"^(>|>=|<|<=|==|!=)$")
    evaluation_window_minutes: int = Field(5, ge=1, le=1440)
    
    # Configuration d'escalade
    escalation_enabled: bool = Field(True)
    escalation_delay_minutes: int = Field(15, ge=1, le=1440)
    max_escalation_level: int = Field(3, ge=1, le=10)
    
    # Suppression et dé-duplication
    suppression_enabled: bool = Field(False)
    suppression_duration_minutes: int = Field(60, ge=1, le=10080)
    deduplication_key_template: Optional[str] = Field(None)
    
    # Notification
    notification_channels: List[NotificationChannel] = Field(default_factory=list)
    notification_template_id: Optional[UUID] = Field(None)
    
    # Validité temporelle
    effective_from: Optional[datetime] = Field(None)
    effective_until: Optional[datetime] = Field(None)
    timezone: str = Field("UTC")
    
    # État
    is_active: bool = Field(True)
    priority: Priority = Field(Priority.NORMAL)
    
    @validator('trigger_condition')
    def validate_trigger_condition(cls, v):
        """Valide la condition de déclenchement"""
        if not v or v.isspace():
            raise ValueError('Trigger condition cannot be empty')
        # Ici on pourrait ajouter une validation plus complexe
        return v.strip()
    
    @root_validator
    def validate_effectiveness_dates(cls, values):
        """Valide les dates d'efficacité"""
        effective_from = values.get('effective_from')
        effective_until = values.get('effective_until')
        
        if effective_from and effective_until:
            if effective_from >= effective_until:
                raise ValueError('effective_from must be before effective_until')
        
        return values
    
    @computed_field
    @property
    def is_currently_effective(self) -> bool:
        """Indique si la règle est actuellement effective"""
        now = datetime.now(timezone.utc)
        
        if self.effective_from and now < self.effective_from:
            return False
        
        if self.effective_until and now > self.effective_until:
            return False
        
        return self.is_active


class AlertInstance(BaseSchema):
    """Instance d'alerte générée par une règle"""
    rule_id: UUID = Field(..., description="ID de la règle qui a généré l'alerte")
    alert_id: StrictStr = Field(..., description="ID unique de l'alerte")
    
    # Contenu de l'alerte
    title: StrictStr = Field(..., min_length=1, max_length=255)
    message: StrictStr = Field(..., min_length=1, max_length=2000)
    level: AlertLevel = Field(...)
    category: WarningCategory = Field(...)
    
    # État et statut
    status: AlertStatus = Field(AlertStatus.PENDING)
    priority: Priority = Field(Priority.NORMAL)
    severity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Contexte
    service_name: Optional[str] = Field(None)
    environment: Environment = Field(Environment.DEVELOPMENT)
    correlation_id: Optional[UUID] = Field(None)
    incident_id: Optional[UUID] = Field(None)
    
    # Données de déclenchement
    trigger_data: Dict[str, Any] = Field(default_factory=dict)
    threshold_exceeded: Optional[float] = Field(None)
    current_value: Optional[float] = Field(None)
    
    # Horodatage
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    first_seen_at: Optional[datetime] = Field(None)
    last_seen_at: Optional[datetime] = Field(None)
    acknowledged_at: Optional[datetime] = Field(None)
    resolved_at: Optional[datetime] = Field(None)
    escalated_at: Optional[datetime] = Field(None)
    
    # Assignation et traitement
    assigned_to: Optional[str] = Field(None)
    acknowledged_by: Optional[str] = Field(None)
    resolved_by: Optional[str] = Field(None)
    escalated_by: Optional[str] = Field(None)
    
    # Escalade
    escalation_level: int = Field(0, ge=0)
    escalation_count: int = Field(0, ge=0)
    escalation_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Suppression
    is_suppressed: bool = Field(False)
    suppressed_until: Optional[datetime] = Field(None)
    suppression_reason: Optional[str] = Field(None)
    
    # Notification
    notification_attempts: List[Dict[str, Any]] = Field(default_factory=list)
    last_notification_sent: Optional[datetime] = Field(None)
    notification_channels_used: List[NotificationChannel] = Field(default_factory=list)
    
    # Groupement et dé-duplication
    deduplication_key: Optional[str] = Field(None)
    duplicate_count: int = Field(0, ge=0)
    parent_alert_id: Optional[UUID] = Field(None)
    child_alert_ids: List[UUID] = Field(default_factory=list)
    
    # Métriques
    time_to_acknowledge_seconds: Optional[int] = Field(None, ge=0)
    time_to_resolve_seconds: Optional[int] = Field(None, ge=0)
    
    @validator('message')
    def validate_message(cls, v):
        """Valide le message d'alerte"""
        if not v or v.isspace():
            raise ValueError('Message cannot be empty')
        return v.strip()
    
    @validator('severity_score', 'confidence_score')
    def validate_scores(cls, v):
        """Valide les scores"""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Score must be between 0.0 and 1.0')
        return v
    
    @root_validator
    def validate_status_transitions(cls, values):
        """Valide les transitions de statut"""
        status = values.get('status')
        acknowledged_at = values.get('acknowledged_at')
        resolved_at = values.get('resolved_at')
        
        if status == AlertStatus.ACKNOWLEDGED and not acknowledged_at:
            values['acknowledged_at'] = datetime.now(timezone.utc)
        
        if status == AlertStatus.RESOLVED and not resolved_at:
            values['resolved_at'] = datetime.now(timezone.utc)
        
        return values
    
    @computed_field
    @property
    def duration_seconds(self) -> Optional[int]:
        """Durée totale de l'alerte en secondes"""
        if not self.triggered_at:
            return None
        
        end_time = self.resolved_at or datetime.now(timezone.utc)
        return int((end_time - self.triggered_at).total_seconds())
    
    @computed_field
    @property
    def is_active(self) -> bool:
        """Indique si l'alerte est active"""
        return self.status.is_active and not self.is_suppressed
    
    @computed_field
    @property
    def urgency_score(self) -> float:
        """Score d'urgence basé sur niveau, priorité et temps"""
        base_score = self.level.priority * 10
        priority_bonus = int(self.priority) * 5
        
        # Bonus temporel (plus c'est vieux, plus c'est urgent)
        if self.triggered_at:
            age_hours = (datetime.now(timezone.utc) - self.triggered_at).total_seconds() / 3600
            time_bonus = min(age_hours * 2, 20)  # Max 20 points
        else:
            time_bonus = 0
        
        return min(base_score + priority_bonus + time_bonus, 100)
    
    def acknowledge(self, user: str, comment: Optional[str] = None) -> bool:
        """Acquitte l'alerte"""
        if self.status not in [AlertStatus.PENDING, AlertStatus.SENT]:
            return False
        
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now(timezone.utc)
        self.acknowledged_by = user
        
        if comment:
            self.metadata.setdefault('comments', []).append({
                'user': user,
                'comment': comment,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action': 'acknowledge'
            })
        
        return True
    
    def resolve(self, user: str, resolution: Optional[str] = None) -> bool:
        """Résout l'alerte"""
        if self.status.is_final:
            return False
        
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)
        self.resolved_by = user
        
        if resolution:
            self.metadata.setdefault('resolution', resolution)
            self.metadata.setdefault('comments', []).append({
                'user': user,
                'comment': resolution,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action': 'resolve'
            })
        
        return True
    
    def escalate(self, user: str, reason: Optional[str] = None) -> bool:
        """Escalade l'alerte"""
        if self.status.is_final:
            return False
        
        self.escalation_level += 1
        self.escalation_count += 1
        self.status = AlertStatus.ESCALATED
        self.escalated_at = datetime.now(timezone.utc)
        self.escalated_by = user
        
        escalation_record = {
            'level': self.escalation_level,
            'escalated_by': user,
            'escalated_at': datetime.now(timezone.utc).isoformat(),
            'reason': reason
        }
        self.escalation_history.append(escalation_record)
        
        return True
    
    def suppress(self, duration_minutes: int, reason: Optional[str] = None) -> bool:
        """Supprime temporairement l'alerte"""
        if self.status.is_final:
            return False
        
        self.is_suppressed = True
        self.suppressed_until = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        self.suppression_reason = reason
        self.status = AlertStatus.SUPPRESSED
        
        return True
    
    def add_notification_attempt(self, channel: NotificationChannel, 
                               success: bool, details: Optional[Dict[str, Any]] = None):
        """Ajoute une tentative de notification"""
        attempt = {
            'channel': channel.value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'success': success,
            'details': details or {}
        }
        
        self.notification_attempts.append(attempt)
        
        if success:
            self.last_notification_sent = datetime.now(timezone.utc)
            if channel not in self.notification_channels_used:
                self.notification_channels_used.append(channel)


class AlertSummary(BaseModel):
    """Résumé d'alerte pour tableaux de bord"""
    alert_id: str = Field(...)
    title: str = Field(...)
    level: AlertLevel = Field(...)
    status: AlertStatus = Field(...)
    category: WarningCategory = Field(...)
    triggered_at: datetime = Field(...)
    tenant_id: str = Field(...)
    service_name: Optional[str] = Field(None)
    environment: Environment = Field(...)
    urgency_score: float = Field(...)
    duration_seconds: Optional[int] = Field(None)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AlertGroup(BaseSchema):
    """Groupe d'alertes pour réduction du bruit"""
    name: StrictStr = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None)
    
    # Critères de groupement
    grouping_criteria: Dict[str, Any] = Field(...)
    group_by_fields: List[str] = Field(default_factory=list)
    
    # Configuration
    max_alerts_in_group: int = Field(100, ge=1, le=1000)
    group_timeout_minutes: int = Field(30, ge=1, le=1440)
    auto_resolve_when_empty: bool = Field(True)
    
    # État du groupe
    alert_count: int = Field(0, ge=0)
    active_alert_count: int = Field(0, ge=0)
    first_alert_at: Optional[datetime] = Field(None)
    last_alert_at: Optional[datetime] = Field(None)
    
    # Alertes membres
    alert_ids: List[UUID] = Field(default_factory=list)
    representative_alert_id: Optional[UUID] = Field(None)
    
    # Notification de groupe
    group_notification_sent: bool = Field(False)
    group_notification_at: Optional[datetime] = Field(None)
    
    @computed_field
    @property
    def is_active(self) -> bool:
        """Indique si le groupe est actif"""
        return self.active_alert_count > 0
    
    def add_alert(self, alert_id: UUID) -> bool:
        """Ajoute une alerte au groupe"""
        if alert_id in self.alert_ids:
            return False
        
        if len(self.alert_ids) >= self.max_alerts_in_group:
            return False
        
        self.alert_ids.append(alert_id)
        self.alert_count += 1
        self.active_alert_count += 1
        self.last_alert_at = datetime.now(timezone.utc)
        
        if not self.first_alert_at:
            self.first_alert_at = self.last_alert_at
        
        if not self.representative_alert_id:
            self.representative_alert_id = alert_id
        
        return True
    
    def remove_alert(self, alert_id: UUID) -> bool:
        """Retire une alerte du groupe"""
        if alert_id not in self.alert_ids:
            return False
        
        self.alert_ids.remove(alert_id)
        self.alert_count = max(0, self.alert_count - 1)
        self.active_alert_count = max(0, self.active_alert_count - 1)
        
        if alert_id == self.representative_alert_id and self.alert_ids:
            self.representative_alert_id = self.alert_ids[0]
        elif not self.alert_ids:
            self.representative_alert_id = None
        
        return True


class AlertFilter(BaseModel):
    """Filtre pour recherche d'alertes"""
    # Filtres temporels
    start_date: Optional[datetime] = Field(None)
    end_date: Optional[datetime] = Field(None)
    last_hours: Optional[int] = Field(None, ge=1, le=8760)  # Max 1 an
    
    # Filtres par attributs
    levels: Optional[List[AlertLevel]] = Field(None)
    statuses: Optional[List[AlertStatus]] = Field(None)
    categories: Optional[List[WarningCategory]] = Field(None)
    priorities: Optional[List[Priority]] = Field(None)
    environments: Optional[List[Environment]] = Field(None)
    
    # Filtres par service
    service_names: Optional[List[str]] = Field(None)
    tenant_ids: Optional[List[str]] = Field(None)
    
    # Filtres par état
    is_acknowledged: Optional[bool] = Field(None)
    is_resolved: Optional[bool] = Field(None)
    is_escalated: Optional[bool] = Field(None)
    is_suppressed: Optional[bool] = Field(None)
    
    # Filtres par score
    min_severity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_severity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_urgency_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    max_urgency_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Recherche textuelle
    search_text: Optional[str] = Field(None, max_length=255)
    
    # Filtres avancés
    has_correlation_id: Optional[bool] = Field(None)
    has_incident_id: Optional[bool] = Field(None)
    escalation_level_min: Optional[int] = Field(None, ge=0)
    escalation_level_max: Optional[int] = Field(None, ge=0)
    
    @root_validator
    def validate_date_range(cls, values):
        """Valide la cohérence des dates"""
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        last_hours = values.get('last_hours')
        
        if start_date and end_date and start_date > end_date:
            raise ValueError('start_date cannot be after end_date')
        
        if last_hours and (start_date or end_date):
            raise ValueError('Cannot use last_hours with start_date or end_date')
        
        return values
    
    @root_validator
    def validate_score_ranges(cls, values):
        """Valide les plages de scores"""
        min_sev = values.get('min_severity_score')
        max_sev = values.get('max_severity_score')
        min_urg = values.get('min_urgency_score')
        max_urg = values.get('max_urgency_score')
        
        if min_sev is not None and max_sev is not None and min_sev > max_sev:
            raise ValueError('min_severity_score cannot be greater than max_severity_score')
        
        if min_urg is not None and max_urg is not None and min_urg > max_urg:
            raise ValueError('min_urgency_score cannot be greater than max_urgency_score')
        
        return values
