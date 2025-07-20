"""
Compliance Schema Module
========================

Ce module définit les schémas pour la gestion de la compliance multi-tenant
avec support GDPR, HIPAA, SOC2, ISO27001, PCI-DSS et autres standards
de conformité avec audit automatisé et reporting avancé.
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.networks import HttpUrl, EmailStr


class ComplianceStandard(str, Enum):
    """Standards de compliance supportés."""
    GDPR = "gdpr"                    # General Data Protection Regulation
    HIPAA = "hipaa"                  # Health Insurance Portability and Accountability Act
    SOC2 = "soc2"                    # Service Organization Control 2
    ISO27001 = "iso27001"            # Information Security Management
    PCI_DSS = "pci_dss"              # Payment Card Industry Data Security Standard
    CCPA = "ccpa"                    # California Consumer Privacy Act
    SOX = "sox"                      # Sarbanes-Oxley Act
    FISMA = "fisma"                  # Federal Information Security Management Act
    NIST = "nist"                    # National Institute of Standards and Technology
    CSF = "csf"                      # Cybersecurity Framework


class ComplianceStatus(str, Enum):
    """États de compliance."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    PENDING_REVIEW = "pending_review"
    IN_PROGRESS = "in_progress"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Niveaux de risque."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataClassification(str, Enum):
    """Classification des données."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class AuditType(str, Enum):
    """Types d'audit."""
    INTERNAL = "internal"
    EXTERNAL = "external"
    SELF_ASSESSMENT = "self_assessment"
    CERTIFICATION = "certification"
    PENETRATION_TEST = "penetration_test"
    VULNERABILITY_SCAN = "vulnerability_scan"


class ControlType(str, Enum):
    """Types de contrôles."""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"
    COMPENSATING = "compensating"
    DIRECTIVE = "directive"


class ComplianceControl(BaseModel):
    """Contrôle de compliance."""
    control_id: str = Field(..., description="ID du contrôle")
    name: str = Field(..., description="Nom du contrôle")
    description: str = Field(..., description="Description détaillée")
    
    # Classification
    control_type: ControlType = Field(..., description="Type de contrôle")
    category: str = Field(..., description="Catégorie")
    subcategory: Optional[str] = Field(None, description="Sous-catégorie")
    
    # Standards associés
    standards: List[ComplianceStandard] = Field(..., description="Standards applicables")
    requirements: List[str] = Field(..., description="Exigences spécifiques")
    
    # État et évaluation
    status: ComplianceStatus = Field(..., description="État de compliance")
    risk_level: RiskLevel = Field(..., description="Niveau de risque")
    last_assessment_date: Optional[datetime] = None
    next_assessment_date: Optional[datetime] = None
    
    # Implémentation
    implementation_status: str = Field(..., regex="^(not_started|in_progress|implemented|verified)$")
    responsible_party: str = Field(..., description="Responsable")
    implementation_date: Optional[datetime] = None
    
    # Documentation
    documentation_links: List[HttpUrl] = Field(default_factory=list)
    evidence_artifacts: List[str] = Field(default_factory=list)
    
    # Tests et validation
    automated_testing: bool = Field(False, description="Tests automatisés")
    testing_frequency: Optional[str] = Field(None, description="Fréquence de test")
    last_test_date: Optional[datetime] = None
    test_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Métriques
    effectiveness_score: Optional[float] = Field(None, ge=0, le=100)
    maturity_level: Optional[int] = Field(None, ge=1, le=5)
    
    class Config:
        schema_extra = {
            "example": {
                "control_id": "AC-001",
                "name": "Access Control Policy",
                "description": "Establish and maintain access control policies",
                "control_type": "preventive",
                "category": "access_control",
                "standards": ["iso27001", "soc2"],
                "requirements": ["Multi-factor authentication", "Role-based access"],
                "status": "compliant",
                "risk_level": "low",
                "implementation_status": "implemented",
                "responsible_party": "security-team@company.com"
            }
        }


class DataRetentionPolicy(BaseModel):
    """Politique de rétention des données."""
    policy_id: str = Field(..., description="ID de la politique")
    name: str = Field(..., description="Nom de la politique")
    data_types: List[str] = Field(..., description="Types de données concernées")
    
    # Classification et sensibilité
    data_classification: DataClassification = Field(..., description="Classification")
    contains_pii: bool = Field(False, description="Contient des données PII")
    contains_phi: bool = Field(False, description="Contient des données PHI")
    
    # Rétention
    retention_period_days: int = Field(..., ge=1, description="Période de rétention (jours)")
    legal_hold_exemption: bool = Field(False, description="Exemption pour conservation légale")
    
    # Suppression
    auto_deletion_enabled: bool = Field(True, description="Suppression automatique")
    deletion_method: str = Field(..., regex="^(soft_delete|hard_delete|anonymization|encryption)$")
    
    # Compliance
    applicable_standards: List[ComplianceStandard] = Field(..., description="Standards applicables")
    legal_basis: List[str] = Field(default_factory=list, description="Base légale")
    
    # Métadonnées
    created_by: str = Field(..., description="Créé par")
    approved_by: Optional[str] = Field(None, description="Approuvé par")
    effective_date: datetime = Field(..., description="Date d'entrée en vigueur")
    review_date: datetime = Field(..., description="Date de révision")
    
    class Config:
        schema_extra = {
            "example": {
                "policy_id": "retention_user_data",
                "name": "User Data Retention Policy",
                "data_types": ["user_profiles", "activity_logs", "preferences"],
                "data_classification": "confidential",
                "contains_pii": True,
                "retention_period_days": 1095,
                "auto_deletion_enabled": True,
                "deletion_method": "anonymization",
                "applicable_standards": ["gdpr", "ccpa"]
            }
        }


class AuditTrail(BaseModel):
    """Piste d'audit pour traçabilité."""
    audit_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Identification
    user_id: str = Field(..., description="ID utilisateur")
    tenant_id: str = Field(..., description="ID tenant")
    session_id: Optional[str] = Field(None, description="ID session")
    
    # Action
    action: str = Field(..., description="Action effectuée")
    resource_type: str = Field(..., description="Type de ressource")
    resource_id: str = Field(..., description="ID de la ressource")
    
    # Détails
    before_state: Optional[Dict[str, Any]] = Field(None, description="État avant")
    after_state: Optional[Dict[str, Any]] = Field(None, description="État après")
    changes: List[Dict[str, Any]] = Field(default_factory=list, description="Changements détaillés")
    
    # Contexte
    ip_address: Optional[str] = Field(None, description="Adresse IP")
    user_agent: Optional[str] = Field(None, description="User Agent")
    source_system: Optional[str] = Field(None, description="Système source")
    
    # Résultat
    success: bool = Field(..., description="Succès de l'action")
    error_message: Optional[str] = Field(None, description="Message d'erreur")
    risk_score: Optional[float] = Field(None, ge=0, le=100, description="Score de risque")
    
    # Métadonnées
    tags: Dict[str, str] = Field(default_factory=dict)
    compliance_relevant: bool = Field(True, description="Pertinent pour compliance")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "tenant_id": "enterprise_001",
                "action": "data_export",
                "resource_type": "user_data",
                "resource_id": "export_456",
                "ip_address": "192.168.1.100",
                "success": True,
                "compliance_relevant": True
            }
        }


class ComplianceAssessment(BaseModel):
    """Évaluation de compliance."""
    assessment_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Nom de l'évaluation")
    description: str = Field(..., description="Description")
    
    # Scope
    tenant_id: str = Field(..., description="ID tenant")
    standards: List[ComplianceStandard] = Field(..., description="Standards évalués")
    scope: List[str] = Field(..., description="Périmètre d'évaluation")
    
    # Type et méthode
    assessment_type: AuditType = Field(..., description="Type d'évaluation")
    methodology: str = Field(..., description="Méthodologie utilisée")
    
    # Planning
    start_date: datetime = Field(..., description="Date de début")
    end_date: datetime = Field(..., description="Date de fin")
    assessor: str = Field(..., description="Évaluateur")
    
    # Résultats
    overall_status: ComplianceStatus = Field(..., description="État global")
    overall_score: float = Field(..., ge=0, le=100, description="Score global")
    controls_evaluated: List[str] = Field(..., description="Contrôles évalués")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="Constats")
    
    # Actions correctives
    action_items: List[Dict[str, Any]] = Field(default_factory=list, description="Actions correctives")
    remediation_deadline: Optional[datetime] = Field(None, description="Échéance de remédiation")
    
    # Documentation
    report_url: Optional[HttpUrl] = Field(None, description="URL du rapport")
    evidence_documents: List[str] = Field(default_factory=list, description="Documents de preuve")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Q4 2024 SOC2 Assessment",
                "description": "Quarterly SOC2 Type II assessment",
                "tenant_id": "enterprise_001",
                "standards": ["soc2"],
                "assessment_type": "external",
                "start_date": "2024-10-01T00:00:00Z",
                "end_date": "2024-10-31T23:59:59Z",
                "overall_status": "compliant",
                "overall_score": 94.5
            }
        }


class IncidentReport(BaseModel):
    """Rapport d'incident de compliance."""
    incident_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(..., description="Titre de l'incident")
    description: str = Field(..., description="Description détaillée")
    
    # Classification
    incident_type: str = Field(..., regex="^(data_breach|unauthorized_access|system_failure|policy_violation)$")
    severity: RiskLevel = Field(..., description="Sévérité")
    affected_standards: List[ComplianceStandard] = Field(..., description="Standards impactés")
    
    # Timing
    discovered_at: datetime = Field(..., description="Date de découverte")
    occurred_at: Optional[datetime] = Field(None, description="Date d'occurrence estimée")
    reported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    
    # Impact
    affected_users: Optional[int] = Field(None, ge=0, description="Utilisateurs affectés")
    affected_records: Optional[int] = Field(None, ge=0, description="Enregistrements affectés")
    data_types_affected: List[str] = Field(default_factory=list, description="Types de données affectées")
    business_impact: Optional[str] = Field(None, description="Impact business")
    
    # Réponse
    containment_actions: List[str] = Field(default_factory=list, description="Actions de confinement")
    remediation_actions: List[str] = Field(default_factory=list, description="Actions de remédiation")
    preventive_measures: List[str] = Field(default_factory=list, description="Mesures préventives")
    
    # Notifications
    regulatory_notification_required: bool = Field(False, description="Notification réglementaire requise")
    regulatory_notifications_sent: List[Dict[str, Any]] = Field(default_factory=list)
    customer_notification_required: bool = Field(False, description="Notification client requise")
    
    # Responsables
    reported_by: str = Field(..., description="Rapporté par")
    assigned_to: str = Field(..., description="Assigné à")
    dpo_notified: bool = Field(False, description="DPO notifié")
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Unauthorized Access to Customer Database",
                "description": "Potential unauthorized access detected in customer database",
                "incident_type": "unauthorized_access",
                "severity": "high",
                "affected_standards": ["gdpr", "iso27001"],
                "discovered_at": "2024-01-15T14:30:00Z",
                "affected_users": 1500,
                "regulatory_notification_required": True,
                "reported_by": "security-team@company.com"
            }
        }


class ComplianceSchema(BaseModel):
    """
    Schéma principal de compliance multi-tenant avec support complet
    des standards internationaux et automatisation des processus d'audit.
    """
    # Identifiants
    compliance_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str = Field(..., description="ID du tenant")
    name: str = Field(..., description="Nom du programme de compliance")
    description: Optional[str] = Field(None, description="Description")
    
    # Standards et frameworks
    applicable_standards: List[ComplianceStandard] = Field(..., description="Standards applicables")
    compliance_framework: str = Field(..., description="Framework de compliance")
    certification_targets: List[str] = Field(default_factory=list, description="Certifications cibles")
    
    # Configuration
    enabled: bool = Field(True, description="Compliance activée")
    monitoring_enabled: bool = Field(True, description="Monitoring activé")
    automated_reporting: bool = Field(True, description="Reporting automatisé")
    
    # Contrôles
    controls: List[ComplianceControl] = Field(default_factory=list, description="Contrôles de compliance")
    control_testing_frequency: str = Field("quarterly", description="Fréquence de test des contrôles")
    
    # Politiques
    data_retention_policies: List[DataRetentionPolicy] = Field(default_factory=list)
    privacy_policies: Dict[str, Any] = Field(default_factory=dict, description="Politiques de confidentialité")
    security_policies: Dict[str, Any] = Field(default_factory=dict, description="Politiques de sécurité")
    
    # Audit et évaluation
    assessments: List[ComplianceAssessment] = Field(default_factory=list, description="Évaluations")
    audit_trails: List[AuditTrail] = Field(default_factory=list, description="Pistes d'audit")
    incident_reports: List[IncidentReport] = Field(default_factory=list, description="Rapports d'incident")
    
    # Responsabilités
    compliance_officer: str = Field(..., description="Responsable compliance")
    data_protection_officer: Optional[str] = Field(None, description="DPO")
    security_officer: Optional[str] = Field(None, description="Responsable sécurité")
    
    # Reporting
    reporting_frequency: str = Field("monthly", regex="^(daily|weekly|monthly|quarterly|yearly)$")
    stakeholders: List[str] = Field(default_factory=list, description="Parties prenantes")
    report_distribution: List[EmailStr] = Field(default_factory=list, description="Distribution des rapports")
    
    # Métriques et KPIs
    compliance_score: float = Field(0, ge=0, le=100, description="Score de compliance global")
    risk_score: float = Field(0, ge=0, le=100, description="Score de risque")
    maturity_level: int = Field(1, ge=1, le=5, description="Niveau de maturité")
    
    # Métadonnées temporelles
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    last_assessment_date: Optional[datetime] = None
    next_assessment_date: Optional[datetime] = None
    
    # Configuration avancée
    risk_tolerance: RiskLevel = Field(RiskLevel.MEDIUM, description="Tolérance au risque")
    automation_level: str = Field("medium", regex="^(low|medium|high)$")
    integration_endpoints: Dict[str, HttpUrl] = Field(default_factory=dict)
    
    # Tags et métadonnées
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "name": "Enterprise Compliance Program",
                "description": "Comprehensive compliance program for enterprise tenant",
                "applicable_standards": ["gdpr", "soc2", "iso27001"],
                "compliance_framework": "NIST CSF",
                "compliance_officer": "compliance@company.com",
                "data_protection_officer": "dpo@company.com",
                "reporting_frequency": "monthly",
                "compliance_score": 92.5,
                "risk_score": 15.3,
                "maturity_level": 4
            }
        }
    
    @validator('updated_at', always=True)
    def set_updated_at(cls, v):
        """Met à jour automatiquement le timestamp."""
        return v or datetime.now(timezone.utc)
    
    @validator('compliance_score', 'risk_score')
    def validate_scores(cls, v):
        """Valide les scores."""
        if not 0 <= v <= 100:
            raise ValueError("Score must be between 0 and 100")
        return v
    
    @validator('maturity_level')
    def validate_maturity(cls, v):
        """Valide le niveau de maturité."""
        if not 1 <= v <= 5:
            raise ValueError("Maturity level must be between 1 and 5")
        return v
    
    @root_validator
    def validate_standards_controls(cls, values):
        """Valide que les contrôles couvrent les standards applicables."""
        applicable_standards = values.get('applicable_standards', [])
        controls = values.get('controls', [])
        
        # Vérifier la couverture des standards
        covered_standards = set()
        for control in controls:
            covered_standards.update(control.standards)
        
        missing_standards = set(applicable_standards) - covered_standards
        if missing_standards:
            raise ValueError(f"Missing controls for standards: {missing_standards}")
        
        return values
    
    @root_validator
    def validate_gdpr_requirements(cls, values):
        """Valide les exigences GDPR si applicable."""
        standards = values.get('applicable_standards', [])
        
        if ComplianceStandard.GDPR in standards:
            dpo = values.get('data_protection_officer')
            if not dpo:
                raise ValueError("GDPR compliance requires a Data Protection Officer")
            
            retention_policies = values.get('data_retention_policies', [])
            if not retention_policies:
                raise ValueError("GDPR compliance requires data retention policies")
        
        return values
    
    def calculate_compliance_score(self) -> float:
        """Calcule le score de compliance basé sur les contrôles."""
        if not self.controls:
            return 0.0
        
        total_weight = 0
        weighted_score = 0
        
        for control in self.controls:
            # Poids basé sur le niveau de risque
            weight = {
                RiskLevel.CRITICAL: 5,
                RiskLevel.HIGH: 4,
                RiskLevel.MEDIUM: 3,
                RiskLevel.LOW: 2,
                RiskLevel.VERY_LOW: 1
            }.get(control.risk_level, 3)
            
            # Score basé sur le statut
            status_score = {
                ComplianceStatus.COMPLIANT: 100,
                ComplianceStatus.PARTIAL: 50,
                ComplianceStatus.NON_COMPLIANT: 0,
                ComplianceStatus.PENDING_REVIEW: 75,
                ComplianceStatus.IN_PROGRESS: 25,
                ComplianceStatus.NOT_APPLICABLE: 100,
                ComplianceStatus.UNKNOWN: 0
            }.get(control.status, 0)
            
            total_weight += weight
            weighted_score += weight * status_score
        
        return (weighted_score / total_weight) if total_weight > 0 else 0.0
    
    def get_non_compliant_controls(self) -> List[ComplianceControl]:
        """Retourne les contrôles non-conformes."""
        return [c for c in self.controls if c.status == ComplianceStatus.NON_COMPLIANT]
    
    def get_high_risk_controls(self) -> List[ComplianceControl]:
        """Retourne les contrôles à haut risque."""
        return [c for c in self.controls 
                if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Génère un rapport de compliance."""
        return {
            "tenant_id": self.tenant_id,
            "report_date": datetime.now(timezone.utc).isoformat(),
            "overall_score": self.compliance_score,
            "risk_score": self.risk_score,
            "maturity_level": self.maturity_level,
            "standards_coverage": {
                standard.value: self._get_standard_coverage(standard)
                for standard in self.applicable_standards
            },
            "controls_summary": {
                "total": len(self.controls),
                "compliant": len([c for c in self.controls if c.status == ComplianceStatus.COMPLIANT]),
                "non_compliant": len(self.get_non_compliant_controls()),
                "high_risk": len(self.get_high_risk_controls())
            },
            "incidents_summary": {
                "total": len(self.incident_reports),
                "open": len([i for i in self.incident_reports if not i.resolved_at]),
                "high_severity": len([i for i in self.incident_reports 
                                    if i.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _get_standard_coverage(self, standard: ComplianceStandard) -> float:
        """Calcule la couverture d'un standard spécifique."""
        relevant_controls = [c for c in self.controls if standard in c.standards]
        if not relevant_controls:
            return 0.0
        
        compliant_controls = [c for c in relevant_controls 
                            if c.status == ComplianceStatus.COMPLIANT]
        return (len(compliant_controls) / len(relevant_controls)) * 100
    
    def _generate_recommendations(self) -> List[str]:
        """Génère des recommandations basées sur l'état actuel."""
        recommendations = []
        
        # Recommandations basées sur les contrôles non-conformes
        non_compliant = self.get_non_compliant_controls()
        if non_compliant:
            recommendations.append(f"Address {len(non_compliant)} non-compliant controls")
        
        # Recommandations basées sur le score
        if self.compliance_score < 80:
            recommendations.append("Improve overall compliance score (currently below 80%)")
        
        # Recommandations basées sur les incidents
        recent_incidents = [i for i in self.incident_reports 
                          if i.reported_at > datetime.now(timezone.utc) - timedelta(days=30)]
        if len(recent_incidents) > 5:
            recommendations.append("Review incident management processes (high incident volume)")
        
        return recommendations
