"""
Schémas d'escalade d'alertes - Spotify AI Agent
Gestion intelligente des politiques d'escalade et de résolution
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Literal
from uuid import UUID, uuid4
from enum import Enum
import json

from pydantic import BaseModel, Field, validator, computed_field, ConfigDict

from . import (
    BaseSchema, TimestampMixin, TenantMixin, MetadataMixin,
    AlertLevel, AlertStatus, WarningCategory, NotificationChannel,
    Priority, Environment
)


class EscalationTrigger(str, Enum):
    """Déclencheurs d'escalade"""
    TIME_BASED = "time_based"              # Basé sur le temps
    ACKNOWLEDGMENT_TIMEOUT = "ack_timeout" # Timeout d'acquittement
    RESOLUTION_TIMEOUT = "resolution_timeout" # Timeout de résolution
    REPEAT_OCCURRENCE = "repeat_occurrence" # Occurrences répétées
    SEVERITY_INCREASE = "severity_increase" # Augmentation de sévérité
    MANUAL = "manual"                      # Escalade manuelle
    AUTO_CORRELATION = "auto_correlation"  # Corrélation automatique


class EscalationAction(str, Enum):
    """Actions d'escalade"""
    NOTIFY_NEXT_LEVEL = "notify_next_level"
    CHANGE_SEVERITY = "change_severity"
    ADD_ASSIGNEE = "add_assignee"
    CREATE_INCIDENT = "create_incident"
    EXECUTE_WEBHOOK = "execute_webhook"
    RUN_AUTOMATION = "run_automation"
    SEND_ALERT_SUMMARY = "send_alert_summary"
    ESCALATE_TO_MANAGER = "escalate_to_manager"


class AssigneeType(str, Enum):
    """Types d'assignés"""
    USER = "user"
    GROUP = "group"
    ROLE = "role"
    ON_CALL_SCHEDULE = "on_call_schedule"
    EXTERNAL_SYSTEM = "external_system"


class EscalationStatus(str, Enum):
    """États d'escalade"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class EscalationLevel(BaseSchema, TimestampMixin):
    """Niveau d'escalade dans une politique"""
    
    level: int = Field(..., ge=1, le=20)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=500)
    
    # Configuration du niveau
    trigger_delay_minutes: int = Field(15, ge=0, le=10080)  # Max 7 jours
    max_attempts: int = Field(3, ge=1, le=10)
    retry_interval_minutes: int = Field(5, ge=1, le=1440)
    
    # Assignés et responsables
    assignees: List[Dict[str, Any]] = Field(default_factory=list)
    backup_assignees: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Actions à exécuter
    actions: List[EscalationAction] = Field(default_factory=list)
    action_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Canaux de notification
    notification_channels: List[UUID] = Field(default_factory=list)
    notification_template_id: Optional[UUID] = Field(None)
    
    # Conditions d'arrêt
    stop_on_acknowledgment: bool = Field(True)
    stop_on_resolution: bool = Field(True)
    auto_resolve_after_hours: Optional[int] = Field(None, ge=1, le=168)  # Max 7 jours
    
    # Configuration avancée
    severity_override: Optional[AlertLevel] = Field(None)
    priority_override: Optional[Priority] = Field(None)
    
    @validator('assignees', 'backup_assignees')
    def validate_assignees(cls, v):
        """Valide la structure des assignés"""
        for assignee in v:
            if 'type' not in assignee or 'id' not in assignee:
                raise ValueError('Assignee must have type and id fields')
            
            if assignee['type'] not in [e.value for e in AssigneeType]:
                raise ValueError(f'Invalid assignee type: {assignee["type"]}')
        
        return v


class EscalationPolicy(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Politique d'escalade complète"""
    
    # Informations de base
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    version: str = Field("1.0.0")
    
    # Configuration globale
    environment: Environment = Field(...)
    enabled: bool = Field(True)
    
    # Niveaux d'escalade
    levels: List[EscalationLevel] = Field(..., min_items=1, max_items=10)
    
    # Déclencheurs
    triggers: List[EscalationTrigger] = Field(default_factory=list)
    trigger_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Conditions d'activation
    alert_level_filter: List[AlertLevel] = Field(default_factory=list)
    category_filter: List[WarningCategory] = Field(default_factory=list)
    priority_filter: List[Priority] = Field(default_factory=list)
    
    # Configuration temporelle
    business_hours_only: bool = Field(False)
    timezone: str = Field("UTC")
    effective_from: Optional[datetime] = Field(None)
    effective_until: Optional[datetime] = Field(None)
    
    # Limites et contrôles
    max_escalations_per_hour: int = Field(100, ge=1, le=1000)
    max_escalations_per_day: int = Field(1000, ge=1, le=10000)
    cooldown_period_minutes: int = Field(60, ge=0, le=1440)
    
    # État et métriques
    total_escalations: int = Field(0, ge=0)
    successful_escalations: int = Field(0, ge=0)
    failed_escalations: int = Field(0, ge=0)
    last_escalation: Optional[datetime] = Field(None)
    
    # Audit et approval
    created_by: Optional[UUID] = Field(None)
    approved_by: Optional[UUID] = Field(None)
    approved_at: Optional[datetime] = Field(None)
    
    # Tags et organisation
    tags: Set[str] = Field(default_factory=set)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    @validator('levels')
    def validate_escalation_levels(cls, v):
        """Valide les niveaux d'escalade"""
        if not v:
            raise ValueError('At least one escalation level is required')
        
        # Vérifier l'unicité des niveaux
        levels = [level.level for level in v]
        if len(levels) != len(set(levels)):
            raise ValueError('Escalation levels must be unique')
        
        # Vérifier l'ordre séquentiel
        sorted_levels = sorted(levels)
        if levels != sorted_levels:
            raise ValueError('Escalation levels must be in sequential order')
        
        return v

    @computed_field
    @property
    def is_active(self) -> bool:
        """Indique si la politique est actuellement active"""
        if not self.enabled:
            return False
        
        now = datetime.now(timezone.utc)
        
        if self.effective_from and now < self.effective_from:
            return False
        
        if self.effective_until and now > self.effective_until:
            return False
        
        return True

    @computed_field
    @property
    def success_rate(self) -> float:
        """Taux de succès des escalades"""
        if self.total_escalations == 0:
            return 0.0
        return (self.successful_escalations / self.total_escalations) * 100

    def get_next_level(self, current_level: int) -> Optional[EscalationLevel]:
        """Obtient le prochain niveau d'escalade"""
        for level in self.levels:
            if level.level > current_level:
                return level
        return None

    def should_escalate(self, alert_data: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Détermine si une escalade doit avoir lieu"""
        if not self.is_active:
            return False
        
        # Vérifier les filtres
        alert_level = alert_data.get('severity')
        if self.alert_level_filter and alert_level not in self.alert_level_filter:
            return False
        
        category = alert_data.get('category')
        if self.category_filter and category not in self.category_filter:
            return False
        
        priority = alert_data.get('priority')
        if self.priority_filter and priority not in self.priority_filter:
            return False
        
        # Vérifier les déclencheurs
        for trigger in self.triggers:
            if self._check_trigger(trigger, alert_data, context):
                return True
        
        return False

    def _check_trigger(self, trigger: EscalationTrigger, 
                      alert_data: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Vérifie un déclencheur spécifique"""
        if trigger == EscalationTrigger.TIME_BASED:
            # Vérifier le timeout basé sur le temps
            created_at = alert_data.get('created_at')
            if created_at:
                elapsed = datetime.now(timezone.utc) - created_at
                timeout = self.trigger_config.get('time_timeout_minutes', 15)
                return elapsed >= timedelta(minutes=timeout)
        
        elif trigger == EscalationTrigger.ACKNOWLEDGMENT_TIMEOUT:
            # Vérifier le timeout d'acquittement
            if alert_data.get('status') != AlertStatus.ACKNOWLEDGED:
                ack_timeout = self.trigger_config.get('ack_timeout_minutes', 30)
                created_at = alert_data.get('created_at')
                if created_at:
                    elapsed = datetime.now(timezone.utc) - created_at
                    return elapsed >= timedelta(minutes=ack_timeout)
        
        elif trigger == EscalationTrigger.REPEAT_OCCURRENCE:
            # Vérifier les occurrences répétées
            repeat_count = context.get('repeat_count', 0)
            threshold = self.trigger_config.get('repeat_threshold', 3)
            return repeat_count >= threshold
        
        elif trigger == EscalationTrigger.SEVERITY_INCREASE:
            # Vérifier l'augmentation de sévérité
            previous_severity = context.get('previous_severity')
            current_severity = alert_data.get('severity')
            if previous_severity and current_severity:
                severity_order = ['debug', 'info', 'warning', 'error', 'critical', 'fatal']
                prev_idx = severity_order.index(previous_severity)
                curr_idx = severity_order.index(current_severity)
                return curr_idx > prev_idx
        
        return False


class EscalationExecution(BaseSchema, TimestampMixin, TenantMixin):
    """Exécution d'une escalade"""
    
    escalation_id: UUID = Field(default_factory=uuid4)
    policy_id: UUID = Field(...)
    alert_id: UUID = Field(...)
    
    # Configuration de l'exécution
    current_level: int = Field(1, ge=1)
    max_level: int = Field(..., ge=1)
    
    # État
    status: EscalationStatus = Field(EscalationStatus.PENDING)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(None)
    
    # Progression
    attempts_count: int = Field(0, ge=0)
    successful_levels: List[int] = Field(default_factory=list)
    failed_levels: List[int] = Field(default_factory=list)
    
    # Détails d'exécution
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)
    last_error: Optional[str] = Field(None)
    
    # Contexte
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @computed_field
    @property
    def duration_minutes(self) -> Optional[float]:
        """Durée de l'escalade en minutes"""
        if not self.completed_at:
            end_time = datetime.now(timezone.utc)
        else:
            end_time = self.completed_at
        
        duration = end_time - self.started_at
        return duration.total_seconds() / 60

    @computed_field
    @property
    def is_completed(self) -> bool:
        """Indique si l'escalade est terminée"""
        return self.status in [
            EscalationStatus.COMPLETED,
            EscalationStatus.FAILED,
            EscalationStatus.CANCELLED
        ]

    def add_log_entry(self, level: int, action: str, result: str, 
                     details: Optional[Dict[str, Any]] = None):
        """Ajoute une entrée au log d'exécution"""
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': level,
            'action': action,
            'result': result,
            'details': details or {}
        }
        self.execution_log.append(entry)

    def mark_level_completed(self, level: int, success: bool):
        """Marque un niveau comme terminé"""
        if success:
            if level not in self.successful_levels:
                self.successful_levels.append(level)
        else:
            if level not in self.failed_levels:
                self.failed_levels.append(level)


class OnCallSchedule(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Planning d'astreinte"""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Configuration du planning
    timezone: str = Field("UTC")
    rotation_type: str = Field("weekly")  # daily, weekly, monthly
    rotation_start: datetime = Field(...)
    
    # Participants
    participants: List[Dict[str, Any]] = Field(..., min_items=1)
    backup_participants: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Règles d'escalade
    escalation_delay_minutes: int = Field(15, ge=1, le=1440)
    max_escalation_attempts: int = Field(3, ge=1, le=10)
    
    # État
    enabled: bool = Field(True)
    current_on_call: Optional[UUID] = Field(None)
    next_rotation: Optional[datetime] = Field(None)
    
    def get_current_on_call(self) -> Optional[Dict[str, Any]]:
        """Obtient la personne actuellement d'astreinte"""
        if not self.participants or not self.enabled:
            return None
        
        # Calcul simplifié - peut être étendu pour des rotations complexes
        now = datetime.now(timezone.utc)
        duration = now - self.rotation_start
        
        if self.rotation_type == "weekly":
            weeks_passed = int(duration.days // 7)
            index = weeks_passed % len(self.participants)
        elif self.rotation_type == "daily":
            days_passed = duration.days
            index = days_passed % len(self.participants)
        else:
            index = 0  # Fallback
        
        return self.participants[index]


__all__ = [
    'EscalationTrigger', 'EscalationAction', 'AssigneeType', 'EscalationStatus',
    'EscalationLevel', 'EscalationPolicy', 'EscalationExecution', 'OnCallSchedule'
]
