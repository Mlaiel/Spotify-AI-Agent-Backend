"""
Schémas de gestion d'incidents - Spotify AI Agent
Gestion complète du cycle de vie des incidents liés aux alertes
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Literal
from uuid import UUID, uuid4
from enum import Enum
import json

from pydantic import BaseModel, Field, validator, computed_field, ConfigDict, EmailStr

from . import (
    BaseSchema, TimestampMixin, TenantMixin, MetadataMixin,
    AlertLevel, AlertStatus, WarningCategory, Priority, Environment
)


class IncidentSeverity(str, Enum):
    """Sévérité d'incident"""
    SEV1 = "sev1"  # Critique - Impact majeur sur le service
    SEV2 = "sev2"  # Élevée - Impact significatif
    SEV3 = "sev3"  # Modérée - Impact mineur
    SEV4 = "sev4"  # Faible - Pas d'impact immédiat
    SEV5 = "sev5"  # Informationnel


class IncidentStatus(str, Enum):
    """États d'incident"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    POSTMORTEM = "postmortem"
    CLOSED = "closed"


class IncidentCategory(str, Enum):
    """Catégories d'incident"""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DATA = "data"
    NETWORK = "network"
    EXTERNAL = "external"
    PLANNED = "planned"


class IncidentImpact(str, Enum):
    """Impact d'incident"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResponseTeamRole(str, Enum):
    """Rôles dans l'équipe de réponse"""
    INCIDENT_COMMANDER = "incident_commander"
    TECHNICAL_LEAD = "technical_lead"
    COMMUNICATIONS_LEAD = "communications_lead"
    SUBJECT_MATTER_EXPERT = "subject_matter_expert"
    OBSERVER = "observer"
    RESPONDER = "responder"


class CommunicationChannel(str, Enum):
    """Canaux de communication"""
    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    PHONE = "phone"
    STATUS_PAGE = "status_page"
    SOCIAL_MEDIA = "social_media"
    INTERNAL_PORTAL = "internal_portal"


class TimelineEventType(str, Enum):
    """Types d'événements de timeline"""
    INCIDENT_CREATED = "incident_created"
    SEVERITY_CHANGED = "severity_changed"
    STATUS_UPDATED = "status_updated"
    TEAM_MEMBER_ADDED = "team_member_added"
    TEAM_MEMBER_REMOVED = "team_member_removed"
    COMMUNICATION_SENT = "communication_sent"
    INVESTIGATION_UPDATE = "investigation_update"
    MITIGATION_APPLIED = "mitigation_applied"
    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    RESOLUTION_IMPLEMENTED = "resolution_implemented"
    POSTMORTEM_SCHEDULED = "postmortem_scheduled"
    LESSON_LEARNED = "lesson_learned"
    INCIDENT_MERGED = "incident_merged"
    INCIDENT_SPLIT = "incident_split"


class ResponseTeamMember(BaseModel):
    """Membre de l'équipe de réponse"""
    
    user_id: UUID = Field(...)
    role: ResponseTeamRole = Field(...)
    joined_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    left_at: Optional[datetime] = Field(None)
    
    # Contact
    email: Optional[EmailStr] = Field(None)
    phone: Optional[str] = Field(None)
    slack_user_id: Optional[str] = Field(None)
    
    # Statut
    available: bool = Field(True)
    on_call: bool = Field(False)
    
    # Notes
    notes: Optional[str] = Field(None, max_length=1000)


class TimelineEvent(BaseModel):
    """Événement de timeline d'incident"""
    
    event_id: UUID = Field(default_factory=uuid4)
    event_type: TimelineEventType = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Détails de l'événement
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    
    # Acteur
    actor_id: Optional[UUID] = Field(None)
    actor_name: Optional[str] = Field(None)
    
    # Données associées
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Visibilité
    public: bool = Field(True)
    internal_only: bool = Field(False)
    
    # Tags et catégories
    tags: Set[str] = Field(default_factory=set)


class IncidentCommunication(BaseModel):
    """Communication d'incident"""
    
    communication_id: UUID = Field(default_factory=uuid4)
    channel: CommunicationChannel = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Contenu
    subject: Optional[str] = Field(None, max_length=255)
    message: str = Field(..., min_length=1)
    
    # Destinataires
    recipients: List[str] = Field(default_factory=list)
    audience: str = Field("internal")  # internal, external, stakeholders, customers
    
    # Métadonnées
    sent_by: Optional[UUID] = Field(None)
    template_used: Optional[UUID] = Field(None)
    
    # Statut de livraison
    delivery_status: str = Field("pending")  # pending, sent, delivered, failed
    delivery_attempts: int = Field(0, ge=0)
    delivery_errors: List[str] = Field(default_factory=list)


class Incident(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Incident complet avec gestion avancée"""
    
    # Informations de base
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1, max_length=5000)
    incident_number: Optional[str] = Field(None)  # Numéro généré automatiquement
    
    # Classification
    severity: IncidentSeverity = Field(...)
    status: IncidentStatus = Field(IncidentStatus.OPEN)
    category: IncidentCategory = Field(...)
    impact: IncidentImpact = Field(...)
    urgency: Priority = Field(...)
    
    # Origine et détection
    detection_method: str = Field("alert")  # alert, monitoring, user_report, manual
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detected_by: Optional[UUID] = Field(None)
    
    # Alertes associées
    related_alert_ids: List[UUID] = Field(default_factory=list)
    primary_alert_id: Optional[UUID] = Field(None)
    correlation_id: Optional[UUID] = Field(None)
    
    # Services et composants affectés
    affected_services: List[str] = Field(default_factory=list)
    affected_components: List[str] = Field(default_factory=list)
    affected_regions: List[str] = Field(default_factory=list)
    affected_environments: List[Environment] = Field(default_factory=list)
    
    # Équipe de réponse
    response_team: List[ResponseTeamMember] = Field(default_factory=list)
    incident_commander: Optional[UUID] = Field(None)
    
    # Timeline et événements
    timeline: List[TimelineEvent] = Field(default_factory=list)
    
    # Communications
    communications: List[IncidentCommunication] = Field(default_factory=list)
    status_page_id: Optional[str] = Field(None)
    war_room_url: Optional[str] = Field(None)
    
    # Résolution
    acknowledged_at: Optional[datetime] = Field(None)
    acknowledged_by: Optional[UUID] = Field(None)
    
    investigating_started_at: Optional[datetime] = Field(None)
    root_cause_identified_at: Optional[datetime] = Field(None)
    mitigation_started_at: Optional[datetime] = Field(None)
    resolved_at: Optional[datetime] = Field(None)
    resolved_by: Optional[UUID] = Field(None)
    
    # Analyse et postmortem
    root_cause: Optional[str] = Field(None, max_length=2000)
    resolution_summary: Optional[str] = Field(None, max_length=2000)
    lessons_learned: List[str] = Field(default_factory=list)
    
    postmortem_required: bool = Field(False)
    postmortem_scheduled_at: Optional[datetime] = Field(None)
    postmortem_completed_at: Optional[datetime] = Field(None)
    postmortem_document_url: Optional[str] = Field(None)
    
    # Métriques et KPI
    time_to_acknowledge_minutes: Optional[float] = Field(None, ge=0)
    time_to_resolve_minutes: Optional[float] = Field(None, ge=0)
    customer_impact_duration_minutes: Optional[float] = Field(None, ge=0)
    
    # Actions de suivi
    follow_up_actions: List[Dict[str, Any]] = Field(default_factory=list)
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Coûts et impact business
    estimated_cost: Optional[float] = Field(None, ge=0)
    revenue_impact: Optional[float] = Field(None)
    customer_count_affected: Optional[int] = Field(None, ge=0)
    
    # Liens externes
    external_ticket_id: Optional[str] = Field(None)
    external_ticket_url: Optional[str] = Field(None)
    runbook_url: Optional[str] = Field(None)
    
    # Tags et labels
    tags: Set[str] = Field(default_factory=set)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    @validator('incident_number', pre=True, always=True)
    def generate_incident_number(cls, v, values):
        """Génère un numéro d'incident unique"""
        if not v:
            # Format: INC-YYYY-MM-DD-XXXX
            now = datetime.now(timezone.utc)
            date_part = now.strftime("%Y-%m-%d")
            # En production, utiliser une séquence ou un compteur
            sequence = f"{now.hour:02d}{now.minute:02d}"
            return f"INC-{date_part}-{sequence}"
        return v

    @computed_field
    @property
    def duration_minutes(self) -> Optional[float]:
        """Durée totale de l'incident en minutes"""
        if not self.resolved_at:
            end_time = datetime.now(timezone.utc)
        else:
            end_time = self.resolved_at
        
        duration = end_time - self.detected_at
        return duration.total_seconds() / 60

    @computed_field
    @property
    def is_active(self) -> bool:
        """Indique si l'incident est encore actif"""
        return self.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]

    @computed_field
    @property
    def current_incident_commander(self) -> Optional[ResponseTeamMember]:
        """Commandant d'incident actuel"""
        for member in self.response_team:
            if (member.role == ResponseTeamRole.INCIDENT_COMMANDER and 
                member.left_at is None):
                return member
        return None

    def add_timeline_event(self, event_type: TimelineEventType, title: str,
                          description: Optional[str] = None, actor_id: Optional[UUID] = None,
                          data: Optional[Dict[str, Any]] = None, public: bool = True):
        """Ajoute un événement à la timeline"""
        event = TimelineEvent(
            event_type=event_type,
            title=title,
            description=description,
            actor_id=actor_id,
            data=data or {},
            public=public
        )
        self.timeline.append(event)
        self.updated_at = datetime.now(timezone.utc)

    def add_team_member(self, user_id: UUID, role: ResponseTeamRole,
                       email: Optional[str] = None, phone: Optional[str] = None):
        """Ajoute un membre à l'équipe de réponse"""
        # Vérifier s'il n'est pas déjà dans l'équipe
        for member in self.response_team:
            if member.user_id == user_id and member.left_at is None:
                return  # Déjà dans l'équipe
        
        member = ResponseTeamMember(
            user_id=user_id,
            role=role,
            email=email,
            phone=phone
        )
        self.response_team.append(member)
        
        # Mettre à jour le commandant d'incident si nécessaire
        if role == ResponseTeamRole.INCIDENT_COMMANDER:
            self.incident_commander = user_id
        
        # Ajouter un événement à la timeline
        self.add_timeline_event(
            TimelineEventType.TEAM_MEMBER_ADDED,
            f"Team member added: {role.value}",
            actor_id=user_id,
            data={'role': role.value, 'user_id': str(user_id)}
        )

    def remove_team_member(self, user_id: UUID):
        """Retire un membre de l'équipe de réponse"""
        for member in self.response_team:
            if member.user_id == user_id and member.left_at is None:
                member.left_at = datetime.now(timezone.utc)
                
                # Ajouter un événement à la timeline
                self.add_timeline_event(
                    TimelineEventType.TEAM_MEMBER_REMOVED,
                    f"Team member removed: {member.role.value}",
                    data={'role': member.role.value, 'user_id': str(user_id)}
                )
                break

    def update_status(self, new_status: IncidentStatus, actor_id: Optional[UUID] = None,
                     notes: Optional[str] = None):
        """Met à jour le statut de l'incident"""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)
        
        # Mettre à jour les timestamps spécifiques
        now = datetime.now(timezone.utc)
        if new_status == IncidentStatus.INVESTIGATING and not self.investigating_started_at:
            self.investigating_started_at = now
        elif new_status == IncidentStatus.RESOLVED and not self.resolved_at:
            self.resolved_at = now
            self.resolved_by = actor_id
            
            # Calculer les métriques
            if self.acknowledged_at:
                ack_duration = self.acknowledged_at - self.detected_at
                self.time_to_acknowledge_minutes = ack_duration.total_seconds() / 60
            
            resolve_duration = now - self.detected_at
            self.time_to_resolve_minutes = resolve_duration.total_seconds() / 60
        
        # Ajouter un événement à la timeline
        self.add_timeline_event(
            TimelineEventType.STATUS_UPDATED,
            f"Status changed from {old_status.value} to {new_status.value}",
            description=notes,
            actor_id=actor_id,
            data={'old_status': old_status.value, 'new_status': new_status.value}
        )

    def acknowledge(self, actor_id: UUID, notes: Optional[str] = None):
        """Acquitte l'incident"""
        if not self.acknowledged_at:
            self.acknowledged_at = datetime.now(timezone.utc)
            self.acknowledged_by = actor_id
            
            # Calculer le temps d'acquittement
            duration = self.acknowledged_at - self.detected_at
            self.time_to_acknowledge_minutes = duration.total_seconds() / 60
            
            # Ajouter un événement à la timeline
            self.add_timeline_event(
                TimelineEventType.INVESTIGATION_UPDATE,
                "Incident acknowledged",
                description=notes,
                actor_id=actor_id
            )

    def add_communication(self, channel: CommunicationChannel, message: str,
                         subject: Optional[str] = None, audience: str = "internal",
                         recipients: Optional[List[str]] = None,
                         sent_by: Optional[UUID] = None):
        """Ajoute une communication"""
        communication = IncidentCommunication(
            channel=channel,
            subject=subject,
            message=message,
            audience=audience,
            recipients=recipients or [],
            sent_by=sent_by
        )
        self.communications.append(communication)
        
        # Ajouter un événement à la timeline
        self.add_timeline_event(
            TimelineEventType.COMMUNICATION_SENT,
            f"Communication sent via {channel.value}",
            description=f"To: {audience}",
            actor_id=sent_by,
            data={'channel': channel.value, 'audience': audience}
        )

    def calculate_sla_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques SLA"""
        metrics = {}
        
        # Time to Acknowledge (TTA)
        if self.acknowledged_at:
            tta = (self.acknowledged_at - self.detected_at).total_seconds() / 60
            metrics['time_to_acknowledge_minutes'] = tta
        
        # Time to Resolve (TTR)
        if self.resolved_at:
            ttr = (self.resolved_at - self.detected_at).total_seconds() / 60
            metrics['time_to_resolve_minutes'] = ttr
        
        # Mean Time to Recovery (MTTR)
        if self.resolved_at and self.investigating_started_at:
            mttr = (self.resolved_at - self.investigating_started_at).total_seconds() / 60
            metrics['mean_time_to_recovery_minutes'] = mttr
        
        # SLA compliance based on severity
        sla_targets = {
            IncidentSeverity.SEV1: {'tta': 15, 'ttr': 240},  # 15 min, 4 hours
            IncidentSeverity.SEV2: {'tta': 30, 'ttr': 480},  # 30 min, 8 hours
            IncidentSeverity.SEV3: {'tta': 60, 'ttr': 1440}, # 1 hour, 24 hours
            IncidentSeverity.SEV4: {'tta': 240, 'ttr': 4320}, # 4 hours, 3 days
        }
        
        target = sla_targets.get(self.severity)
        if target:
            if 'time_to_acknowledge_minutes' in metrics:
                metrics['tta_sla_met'] = metrics['time_to_acknowledge_minutes'] <= target['tta']
            if 'time_to_resolve_minutes' in metrics:
                metrics['ttr_sla_met'] = metrics['time_to_resolve_minutes'] <= target['ttr']
        
        return metrics


class IncidentMetrics(BaseSchema, TimestampMixin, TenantMixin):
    """Métriques d'incidents"""
    
    metrics_id: UUID = Field(default_factory=uuid4)
    period_start: datetime = Field(...)
    period_end: datetime = Field(...)
    
    # Métriques générales
    total_incidents: int = Field(0, ge=0)
    open_incidents: int = Field(0, ge=0)
    resolved_incidents: int = Field(0, ge=0)
    
    # Par sévérité
    incidents_by_severity: Dict[str, int] = Field(default_factory=dict)
    
    # Métriques temporelles
    avg_time_to_acknowledge_minutes: Optional[float] = Field(None, ge=0)
    avg_time_to_resolve_minutes: Optional[float] = Field(None, ge=0)
    median_time_to_resolve_minutes: Optional[float] = Field(None, ge=0)
    
    # SLA compliance
    sla_compliance_rate: Optional[float] = Field(None, ge=0, le=100)
    sev1_sla_compliance: Optional[float] = Field(None, ge=0, le=100)
    sev2_sla_compliance: Optional[float] = Field(None, ge=0, le=100)
    
    # Tendances
    incident_trend: str = Field("stable")  # increasing, decreasing, stable
    resolution_time_trend: str = Field("stable")
    
    # Top causes
    top_root_causes: List[Dict[str, Any]] = Field(default_factory=list)
    top_affected_services: List[Dict[str, Any]] = Field(default_factory=list)


__all__ = [
    'IncidentSeverity', 'IncidentStatus', 'IncidentCategory', 'IncidentImpact',
    'ResponseTeamRole', 'CommunicationChannel', 'TimelineEventType',
    'ResponseTeamMember', 'TimelineEvent', 'IncidentCommunication',
    'Incident', 'IncidentMetrics'
]
