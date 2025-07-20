"""
Enterprise Compliance and Regulatory Schemas
Comprehensive compliance framework for multi-tenant SaaS platform.
"""

from datetime import datetime, timedelta, date
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, EmailStr
from decimal import Decimal
import uuid


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    SOC2 = "soc2"
    FISMA = "fisma"
    PIPEDA = "pipeda"


class DataClassification(str, Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"


class ProcessingLawfulBasis(str, Enum):
    """GDPR lawful basis for processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataSubjectRights(str, Enum):
    """Data subject rights under GDPR."""
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    RESTRICT_PROCESSING = "restrict_processing"
    DATA_PORTABILITY = "data_portability"
    OBJECT_PROCESSING = "object_processing"
    AUTOMATED_DECISION = "automated_decision"


class BreachSeverity(str, Enum):
    """Data breach severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatusSchema(BaseModel):
    """Compliance status tracking."""
    
    framework: ComplianceFramework
    status: str = Field(..., regex=r"^(compliant|non_compliant|partial|under_review)$")
    compliance_percentage: float = Field(..., ge=0.0, le=100.0)
    last_assessment_date: date
    next_assessment_date: date
    
    # Assessment details
    assessed_by: str = Field(..., description="Assessor name or organization")
    assessment_scope: List[str] = Field(..., description="Areas assessed")
    findings: List[str] = Field(default_factory=list)
    remediation_items: List[str] = Field(default_factory=list)
    
    # Certification
    certification_valid: bool = False
    certificate_number: Optional[str] = None
    certificate_expiry: Optional[date] = None
    certification_body: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DataGovernanceSchema(BaseModel):
    """Comprehensive data governance configuration."""
    
    # Data classification
    auto_classification_enabled: bool = True
    classification_rules: List[Dict[str, Any]] = Field(default_factory=list)
    default_classification: DataClassification = DataClassification.INTERNAL
    
    # Data lifecycle management
    data_retention_policies: Dict[str, int] = Field(default_factory=dict)  # days
    automatic_deletion_enabled: bool = True
    archival_policies: Dict[str, int] = Field(default_factory=dict)  # days before archival
    
    # Data lineage
    lineage_tracking_enabled: bool = True
    data_flow_monitoring: bool = True
    impact_analysis_enabled: bool = True
    
    # Data quality
    quality_monitoring_enabled: bool = True
    quality_rules: List[Dict[str, Any]] = Field(default_factory=list)
    data_profiling_enabled: bool = True
    
    # Access governance
    data_access_reviews_enabled: bool = True
    access_review_frequency_days: int = Field(90, ge=30, le=365)
    privileged_access_monitoring: bool = True
    
    # Data discovery
    sensitive_data_discovery: bool = True
    pii_detection_enabled: bool = True
    phi_detection_enabled: bool = True
    financial_data_detection: bool = True


class PrivacyControlsSchema(BaseModel):
    """Privacy controls and data subject rights management."""
    
    # Consent management
    consent_management_enabled: bool = True
    granular_consent: bool = True
    consent_versioning: bool = True
    consent_withdrawal_enabled: bool = True
    
    # Data subject rights
    rights_request_portal: bool = True
    automated_fulfillment: bool = True
    fulfillment_sla_days: int = Field(30, ge=1, le=30)
    
    # Privacy by design
    privacy_impact_assessments: bool = True
    data_minimization_enabled: bool = True
    purpose_limitation_enforced: bool = True
    storage_limitation_enforced: bool = True
    
    # Anonymization and pseudonymization
    anonymization_enabled: bool = True
    pseudonymization_enabled: bool = True
    k_anonymity_level: int = Field(5, ge=2, le=20)
    
    # Cross-border transfers
    adequacy_decisions_respected: bool = True
    standard_contractual_clauses: bool = True
    binding_corporate_rules: bool = False
    
    # Cookies and tracking
    cookie_consent_banner: bool = True
    tracking_consent_required: bool = True
    do_not_track_honored: bool = True


class AuditTrailSchema(BaseModel):
    """Comprehensive audit trail configuration."""
    
    # Audit scope
    all_data_access_logged: bool = True
    administrative_actions_logged: bool = True
    system_changes_logged: bool = True
    security_events_logged: bool = True
    
    # Audit details
    user_identification_required: bool = True
    timestamp_precision: str = "microseconds"
    geolocation_tracking: bool = True
    device_fingerprinting: bool = True
    
    # Integrity protection
    log_encryption_enabled: bool = True
    log_signing_enabled: bool = True
    tamper_detection_enabled: bool = True
    immutable_storage: bool = True
    
    # Retention and archival
    audit_retention_years: int = Field(7, ge=1, le=50)
    long_term_archival: bool = True
    legal_hold_capability: bool = True
    
    # Monitoring and alerting
    suspicious_activity_detection: bool = True
    compliance_violation_alerts: bool = True
    automated_reporting: bool = True


class BreachManagementSchema(BaseModel):
    """Data breach management and response."""
    
    # Detection
    automated_breach_detection: bool = True
    anomaly_detection_enabled: bool = True
    real_time_monitoring: bool = True
    
    # Response procedures
    incident_response_plan: bool = True
    automated_containment: bool = True
    forensic_preservation: bool = True
    
    # Notification requirements
    regulatory_notification_enabled: bool = True
    notification_sla_hours: int = Field(72, ge=1, le=72)
    data_subject_notification: bool = True
    public_disclosure_threshold: int = Field(500, ge=1)
    
    # Documentation
    breach_register_maintained: bool = True
    impact_assessment_required: bool = True
    lessons_learned_documented: bool = True


class RegulatoryReportingSchema(BaseModel):
    """Regulatory reporting and documentation."""
    
    # Automated reporting
    automated_compliance_reports: bool = True
    scheduled_report_generation: bool = True
    real_time_compliance_dashboard: bool = True
    
    # Report types
    privacy_impact_assessments: bool = True
    data_protection_impact_assessments: bool = True
    records_of_processing_activities: bool = True
    transfer_impact_assessments: bool = True
    
    # Documentation management
    policy_version_control: bool = True
    procedure_documentation: bool = True
    training_records_maintained: bool = True
    vendor_due_diligence_records: bool = True
    
    # External reporting
    regulatory_filing_automation: bool = True
    third_party_audit_support: bool = True
    certification_maintenance: bool = True


class ComplianceConfigSchema(BaseModel):
    """Main compliance configuration schema."""
    
    tenant_id: str = Field(..., description="Tenant identifier")
    organization_name: str = Field(..., min_length=2, max_length=200)
    jurisdiction: str = Field(..., description="Primary legal jurisdiction")
    
    # Applicable frameworks
    applicable_frameworks: List[ComplianceFramework] = Field(..., min_items=1)
    framework_status: List[ComplianceStatusSchema] = Field(default_factory=list)
    
    # Core components
    data_governance: DataGovernanceSchema
    privacy_controls: PrivacyControlsSchema
    audit_trail: AuditTrailSchema
    breach_management: BreachManagementSchema
    regulatory_reporting: RegulatoryReportingSchema
    
    # Organization details
    data_protection_officer: Optional[EmailStr] = None
    privacy_officer: Optional[EmailStr] = None
    compliance_officer: Optional[EmailStr] = None
    legal_contact: Optional[EmailStr] = None
    
    # Third-party relationships
    data_processors: List[Dict[str, Any]] = Field(default_factory=list)
    data_controllers: List[Dict[str, Any]] = Field(default_factory=list)
    vendor_compliance_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Risk management
    compliance_risk_assessment: bool = True
    risk_mitigation_plans: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_monitoring_enabled: bool = True
    
    # Training and awareness
    compliance_training_required: bool = True
    training_frequency_months: int = Field(12, ge=3, le=24)
    awareness_programs_enabled: bool = True
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('framework_status')
    def validate_framework_status(cls, v, values):
        if 'applicable_frameworks' in values:
            framework_names = [status.framework for status in v]
            for framework in values['applicable_frameworks']:
                if framework not in framework_names:
                    # Auto-create status for missing frameworks
                    v.append(ComplianceStatusSchema(
                        framework=framework,
                        status="under_review",
                        compliance_percentage=0.0,
                        last_assessment_date=date.today(),
                        next_assessment_date=date.today() + timedelta(days=90),
                        assessed_by="System",
                        assessment_scope=["Initial setup"]
                    ))
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "tenant_001",
                "organization_name": "Acme Corporation",
                "jurisdiction": "EU",
                "applicable_frameworks": ["gdpr", "iso_27001"],
                "data_protection_officer": "dpo@acme.com"
            }
        }


class ComplianceAssessmentSchema(BaseModel):
    """Compliance assessment and audit results."""
    
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant identifier")
    framework: ComplianceFramework
    
    # Assessment metadata
    assessment_type: str = Field(..., regex=r"^(self|internal|external|regulatory)$")
    assessor_name: str = Field(..., description="Lead assessor name")
    assessment_firm: Optional[str] = None
    assessment_scope: List[str] = Field(..., description="Areas covered")
    
    # Timeline
    assessment_start_date: date
    assessment_end_date: date
    report_date: date
    
    # Results
    overall_score: float = Field(..., ge=0.0, le=100.0)
    compliance_level: str = Field(..., regex=r"^(compliant|partially_compliant|non_compliant)$")
    
    # Detailed findings
    control_assessments: List[Dict[str, Any]] = Field(default_factory=list)
    gaps_identified: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Risk assessment
    identified_risks: List[Dict[str, Any]] = Field(default_factory=list)
    risk_mitigation_plan: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Follow-up
    remediation_deadline: Optional[date] = None
    next_assessment_date: Optional[date] = None
    certification_recommendation: bool = False
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "tenant_001",
                "framework": "gdpr",
                "assessment_type": "external",
                "assessor_name": "Jane Compliance",
                "overall_score": 85.5,
                "compliance_level": "compliant"
            }
        }
