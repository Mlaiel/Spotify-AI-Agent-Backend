"""
Schémas de conformité et audit - Spotify AI Agent
Gestion complète de la conformité réglementaire et de l'audit
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Literal
from uuid import UUID, uuid4
from enum import Enum
import json
from decimal import Decimal

from pydantic import BaseModel, Field, validator, computed_field, ConfigDict, EmailStr

from . import (
    BaseSchema, TimestampMixin, TenantMixin, MetadataMixin,
    AlertLevel, AlertStatus, WarningCategory, Priority, Environment
)


class ComplianceStandard(str, Enum):
    """Standards de conformité"""
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOX = "sox"
    NIST = "nist"
    COBIT = "cobit"
    ITIL = "itil"
    CUSTOM = "custom"


class AuditType(str, Enum):
    """Types d'audit"""
    INTERNAL = "internal"
    EXTERNAL = "external"
    REGULATORY = "regulatory"
    SECURITY = "security"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"


class AuditStatus(str, Enum):
    """États d'audit"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    EVIDENCE_REVIEW = "evidence_review"
    DRAFT_REPORT = "draft_report"
    FINAL_REPORT = "final_report"
    COMPLETED = "completed"
    FOLLOW_UP = "follow_up"


class ComplianceStatus(str, Enum):
    """États de conformité"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    REMEDIATION_REQUIRED = "remediation_required"
    NOT_APPLICABLE = "not_applicable"


class RiskLevel(str, Enum):
    """Niveaux de risque"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class EvidenceType(str, Enum):
    """Types de preuves"""
    DOCUMENT = "document"
    SCREENSHOT = "screenshot"
    LOG_FILE = "log_file"
    CONFIGURATION = "configuration"
    POLICY = "policy"
    PROCEDURE = "procedure"
    TRAINING_RECORD = "training_record"
    ACCESS_LOG = "access_log"
    SYSTEM_OUTPUT = "system_output"
    INTERVIEW_NOTES = "interview_notes"
    ATTESTATION = "attestation"


class RemediationPriority(str, Enum):
    """Priorités de remédiation"""
    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    PLANNING = "planning"


class ComplianceRequirement(BaseModel):
    """Exigence de conformité"""
    
    requirement_id: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1, max_length=2000)
    
    # Classification
    standard: ComplianceStandard = Field(...)
    category: str = Field(..., min_length=1, max_length=100)
    subcategory: Optional[str] = Field(None, max_length=100)
    
    # Détails de l'exigence
    control_objective: str = Field(..., min_length=1, max_length=500)
    control_activities: List[str] = Field(default_factory=list)
    
    # Évaluation
    risk_level: RiskLevel = Field(...)
    mandatory: bool = Field(True)
    
    # Références
    source_document: Optional[str] = Field(None, max_length=255)
    section_reference: Optional[str] = Field(None, max_length=100)
    related_requirements: List[str] = Field(default_factory=list)
    
    # Métadonnées
    tags: Set[str] = Field(default_factory=set)
    effective_date: Optional[datetime] = Field(None)
    review_frequency_days: Optional[int] = Field(None, ge=1)


class AuditEvidence(BaseModel):
    """Preuve d'audit"""
    
    evidence_id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Type et format
    evidence_type: EvidenceType = Field(...)
    format: str = Field(..., min_length=1, max_length=50)
    
    # Contenu
    file_path: Optional[str] = Field(None)
    file_size_bytes: Optional[int] = Field(None, ge=0)
    checksum: Optional[str] = Field(None)
    content_preview: Optional[str] = Field(None, max_length=1000)
    
    # Métadonnées
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    collected_by: Optional[UUID] = Field(None)
    source_system: Optional[str] = Field(None, max_length=100)
    
    # Validation
    validated: bool = Field(False)
    validated_by: Optional[UUID] = Field(None)
    validated_at: Optional[datetime] = Field(None)
    
    # Rétention
    retention_period_days: Optional[int] = Field(None, ge=1)
    deletion_date: Optional[datetime] = Field(None)
    
    # Classification
    confidentiality_level: str = Field("internal")  # public, internal, confidential, restricted
    tags: Set[str] = Field(default_factory=set)


class ComplianceAssessment(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Évaluation de conformité"""
    
    assessment_id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Configuration
    standard: ComplianceStandard = Field(...)
    scope: str = Field(..., min_length=1, max_length=500)
    period_start: datetime = Field(...)
    period_end: datetime = Field(...)
    
    # Exigences évaluées
    requirements: List[ComplianceRequirement] = Field(default_factory=list)
    requirement_assessments: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Résultats globaux
    overall_status: ComplianceStatus = Field(ComplianceStatus.UNDER_REVIEW)
    compliance_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Métriques
    total_requirements: int = Field(0, ge=0)
    compliant_requirements: int = Field(0, ge=0)
    non_compliant_requirements: int = Field(0, ge=0)
    partially_compliant_requirements: int = Field(0, ge=0)
    
    # Équipe d'évaluation
    lead_assessor: Optional[UUID] = Field(None)
    assessors: List[UUID] = Field(default_factory=list)
    
    # Documentation
    evidence_collected: List[UUID] = Field(default_factory=list)  # Evidence IDs
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # États et dates
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    # Rapport
    report_generated: bool = Field(False)
    report_path: Optional[str] = Field(None)
    executive_summary: Optional[str] = Field(None, max_length=2000)
    
    # Suivi
    next_assessment_date: Optional[datetime] = Field(None)
    remediation_plan_id: Optional[UUID] = Field(None)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    @computed_field
    @property
    def compliance_percentage(self) -> float:
        """Pourcentage de conformité"""
        if self.total_requirements == 0:
            return 0.0
        return (self.compliant_requirements / self.total_requirements) * 100

    @computed_field
    @property
    def assessment_duration_days(self) -> Optional[int]:
        """Durée de l'évaluation en jours"""
        if not self.started_at or not self.completed_at:
            return None
        
        duration = self.completed_at - self.started_at
        return duration.days

    def assess_requirement(self, requirement_id: str, status: ComplianceStatus,
                          evidence_ids: List[UUID], notes: Optional[str] = None,
                          assessor_id: Optional[UUID] = None):
        """Évalue une exigence spécifique"""
        assessment = {
            'requirement_id': requirement_id,
            'status': status.value,
            'evidence_ids': [str(eid) for eid in evidence_ids],
            'notes': notes,
            'assessed_by': str(assessor_id) if assessor_id else None,
            'assessed_at': datetime.now(timezone.utc).isoformat()
        }
        
        self.requirement_assessments[requirement_id] = assessment
        
        # Mettre à jour les compteurs
        self._update_compliance_metrics()

    def _update_compliance_metrics(self):
        """Met à jour les métriques de conformité"""
        self.total_requirements = len(self.requirement_assessments)
        self.compliant_requirements = sum(
            1 for assessment in self.requirement_assessments.values()
            if assessment['status'] == ComplianceStatus.COMPLIANT.value
        )
        self.non_compliant_requirements = sum(
            1 for assessment in self.requirement_assessments.values()
            if assessment['status'] == ComplianceStatus.NON_COMPLIANT.value
        )
        self.partially_compliant_requirements = sum(
            1 for assessment in self.requirement_assessments.values()
            if assessment['status'] == ComplianceStatus.PARTIALLY_COMPLIANT.value
        )
        
        # Calculer le score global
        if self.total_requirements > 0:
            self.compliance_score = (
                (self.compliant_requirements + 0.5 * self.partially_compliant_requirements) /
                self.total_requirements
            ) * 100


class AuditTrail(BaseSchema, TimestampMixin, TenantMixin):
    """Piste d'audit"""
    
    trail_id: UUID = Field(default_factory=uuid4)
    
    # Événement audité
    event_type: str = Field(..., min_length=1, max_length=100)
    event_category: str = Field(..., min_length=1, max_length=50)
    event_description: str = Field(..., min_length=1, max_length=500)
    
    # Acteur
    user_id: Optional[UUID] = Field(None)
    user_email: Optional[EmailStr] = Field(None)
    user_role: Optional[str] = Field(None, max_length=100)
    session_id: Optional[str] = Field(None)
    
    # Contexte technique
    ip_address: Optional[str] = Field(None)
    user_agent: Optional[str] = Field(None)
    source_system: Optional[str] = Field(None, max_length=100)
    
    # Objet de l'action
    resource_type: str = Field(..., min_length=1, max_length=100)
    resource_id: str = Field(..., min_length=1, max_length=255)
    resource_name: Optional[str] = Field(None, max_length=255)
    
    # Détails de l'action
    action: str = Field(..., min_length=1, max_length=100)  # CREATE, READ, UPDATE, DELETE, etc.
    outcome: str = Field(..., min_length=1, max_length=20)  # SUCCESS, FAILURE, ERROR
    
    # Données de l'événement
    before_values: Optional[Dict[str, Any]] = Field(None)
    after_values: Optional[Dict[str, Any]] = Field(None)
    request_data: Optional[Dict[str, Any]] = Field(None)
    response_data: Optional[Dict[str, Any]] = Field(None)
    
    # Métadonnées
    correlation_id: Optional[UUID] = Field(None)
    alert_id: Optional[UUID] = Field(None)
    incident_id: Optional[UUID] = Field(None)
    
    # Classification de sécurité
    security_relevant: bool = Field(False)
    compliance_relevant: bool = Field(False)
    privacy_relevant: bool = Field(False)
    
    # Rétention
    retention_period_days: int = Field(2555, ge=1)  # 7 ans par défaut
    
    # Tags et indexation
    tags: Set[str] = Field(default_factory=set)
    
    @computed_field
    @property
    def is_sensitive(self) -> bool:
        """Indique si l'événement est sensible"""
        return self.security_relevant or self.compliance_relevant or self.privacy_relevant


class RemediationAction(BaseModel):
    """Action de remédiation"""
    
    action_id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1, max_length=1000)
    
    # Classification
    priority: RemediationPriority = Field(...)
    risk_level: RiskLevel = Field(...)
    
    # Assignation
    assigned_to: Optional[UUID] = Field(None)
    assigned_team: Optional[str] = Field(None, max_length=100)
    
    # Échéances
    due_date: Optional[datetime] = Field(None)
    estimated_effort_hours: Optional[float] = Field(None, ge=0)
    
    # Suivi
    status: str = Field("open")  # open, in_progress, completed, cancelled
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    
    # Détails d'implémentation
    implementation_plan: Optional[str] = Field(None, max_length=2000)
    acceptance_criteria: List[str] = Field(default_factory=list)
    
    # Résultats
    completed_at: Optional[datetime] = Field(None)
    verification_evidence: List[UUID] = Field(default_factory=list)
    effectiveness_verified: bool = Field(False)
    
    # Coût et ressources
    estimated_cost: Optional[Decimal] = Field(None, ge=0)
    actual_cost: Optional[Decimal] = Field(None, ge=0)
    
    # Liens
    related_finding_id: Optional[UUID] = Field(None)
    related_requirement_id: Optional[str] = Field(None)


class ComplianceReport(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Rapport de conformité"""
    
    report_id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=1, max_length=255)
    report_type: str = Field(..., min_length=1, max_length=100)
    
    # Période couverte
    reporting_period_start: datetime = Field(...)
    reporting_period_end: datetime = Field(...)
    
    # Standards couverts
    standards: List[ComplianceStandard] = Field(default_factory=list)
    scope: str = Field(..., min_length=1, max_length=1000)
    
    # Métadonnées du rapport
    generated_by: Optional[UUID] = Field(None)
    approved_by: Optional[UUID] = Field(None)
    approved_at: Optional[datetime] = Field(None)
    
    # Contenu du rapport
    executive_summary: str = Field(..., min_length=1, max_length=5000)
    detailed_findings: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Métriques globales
    overall_compliance_score: float = Field(..., ge=0.0, le=100.0)
    total_requirements_assessed: int = Field(..., ge=0)
    compliant_percentage: float = Field(..., ge=0.0, le=100.0)
    
    # Tendances
    compliance_trend: str = Field("stable")  # improving, declining, stable
    previous_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Distribution des risques
    risk_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Actions de remédiation
    total_remediation_actions: int = Field(0, ge=0)
    completed_remediation_actions: int = Field(0, ge=0)
    overdue_remediation_actions: int = Field(0, ge=0)
    
    # Fichiers associés
    report_file_path: Optional[str] = Field(None)
    appendices: List[str] = Field(default_factory=list)
    
    # Distribution
    recipients: List[str] = Field(default_factory=list)
    confidentiality_level: str = Field("confidential")
    
    # État
    status: str = Field("draft")  # draft, review, approved, published
    
    @computed_field
    @property
    def compliance_improvement(self) -> Optional[float]:
        """Amélioration de conformité par rapport au rapport précédent"""
        if self.previous_score is None:
            return None
        return self.overall_compliance_score - self.previous_score

    @computed_field
    @property
    def remediation_completion_rate(self) -> float:
        """Taux de completion des actions de remédiation"""
        if self.total_remediation_actions == 0:
            return 0.0
        return (self.completed_remediation_actions / self.total_remediation_actions) * 100


class ComplianceMonitoring(BaseSchema, TimestampMixin, TenantMixin):
    """Surveillance continue de conformité"""
    
    monitoring_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Configuration
    standard: ComplianceStandard = Field(...)
    monitored_requirements: List[str] = Field(default_factory=list)
    
    # Surveillance automatique
    automated_checks: List[Dict[str, Any]] = Field(default_factory=list)
    check_frequency_hours: int = Field(24, ge=1, le=8760)  # Max 1 an
    
    # Alertes de conformité
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    notification_channels: List[str] = Field(default_factory=list)
    
    # État
    enabled: bool = Field(True)
    last_check: Optional[datetime] = Field(None)
    next_check: Optional[datetime] = Field(None)
    
    # Résultats
    current_compliance_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    compliance_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Violations détectées
    active_violations: List[Dict[str, Any]] = Field(default_factory=list)
    violation_count: int = Field(0, ge=0)
    
    def add_compliance_check(self, score: float, details: Dict[str, Any]):
        """Ajoute un résultat de vérification de conformité"""
        check_result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'score': score,
            'details': details
        }
        
        self.compliance_history.append(check_result)
        self.current_compliance_score = score
        self.last_check = datetime.now(timezone.utc)
        
        # Limiter l'historique (garder les 1000 derniers)
        if len(self.compliance_history) > 1000:
            self.compliance_history = self.compliance_history[-1000:]


__all__ = [
    'ComplianceStandard', 'AuditType', 'AuditStatus', 'ComplianceStatus',
    'RiskLevel', 'EvidenceType', 'RemediationPriority',
    'ComplianceRequirement', 'AuditEvidence', 'ComplianceAssessment',
    'AuditTrail', 'RemediationAction', 'ComplianceReport', 'ComplianceMonitoring'
]
