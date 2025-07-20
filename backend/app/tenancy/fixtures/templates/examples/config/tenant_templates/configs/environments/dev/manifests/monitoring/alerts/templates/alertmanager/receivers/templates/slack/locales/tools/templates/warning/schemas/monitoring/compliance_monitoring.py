"""
Advanced Compliance Monitoring - Industrial Grade Regulatory Compliance System
=============================================================================

Ce module fournit une architecture de monitoring de conformité ultra-avancée
pour la surveillance réglementaire en temps réel avec audit automatisé.

Features:
- Multi-framework compliance monitoring (GDPR, SOC2, ISO27001, HIPAA, PCI-DSS)
- Real-time compliance scoring and reporting
- Automated audit trail generation
- Risk assessment and mitigation tracking
- Policy violation detection and remediation
- Continuous compliance validation
- Regulatory change impact analysis
"""

from typing import Dict, List, Optional, Union, Any, Set
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import json
from uuid import uuid4


class ComplianceFramework(str, Enum):
    """Frameworks de conformité supportés"""
    GDPR = "gdpr"                    # General Data Protection Regulation
    SOC2 = "soc2"                   # Service Organization Control 2
    ISO27001 = "iso27001"           # Information Security Management
    HIPAA = "hipaa"                 # Health Insurance Portability
    PCI_DSS = "pci_dss"             # Payment Card Industry Data Security
    CCPA = "ccpa"                   # California Consumer Privacy Act
    NIST = "nist"                   # NIST Cybersecurity Framework
    CIS = "cis"                     # Center for Internet Security
    FISMA = "fisma"                 # Federal Information Security Management
    FEDRAMP = "fedramp"             # Federal Risk and Authorization Management


class ComplianceStatus(str, Enum):
    """Statuts de conformité"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Niveaux de risque"""
    CRITICAL = "critical"           # Risque critique - action immédiate
    HIGH = "high"                   # Risque élevé - action dans 24h
    MEDIUM = "medium"               # Risque moyen - action dans 7 jours
    LOW = "low"                     # Risque faible - suivi requis
    MINIMAL = "minimal"             # Risque minimal - information


class ViolationType(str, Enum):
    """Types de violations"""
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_RETENTION = "data_retention"
    ENCRYPTION = "encryption"
    ACCESS_CONTROL = "access_control"
    AUDIT_LOGGING = "audit_logging"
    DATA_TRANSFER = "data_transfer"
    PRIVACY_POLICY = "privacy_policy"
    CONSENT_MANAGEMENT = "consent_management"
    INCIDENT_RESPONSE = "incident_response"


class RemediationStatus(str, Enum):
    """Statuts de remédiation"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class ComplianceControl(BaseModel):
    """Contrôle de conformité individuel"""
    
    # Identifiants
    control_id: str = Field(..., description="ID unique du contrôle")
    framework: ComplianceFramework = Field(..., description="Framework de conformité")
    control_number: str = Field(..., description="Numéro de contrôle (ex: SOC2-CC6.1)")
    
    # Description
    title: str = Field(..., description="Titre du contrôle")
    description: str = Field(..., description="Description détaillée")
    objective: str = Field(..., description="Objectif du contrôle")
    
    # Classification
    category: str = Field(..., description="Catégorie (ex: Access Control)")
    subcategory: str = Field("", description="Sous-catégorie")
    
    # Criticité et risque
    criticality: RiskLevel = Field(..., description="Criticité du contrôle")
    inherent_risk: RiskLevel = Field(..., description="Risque inhérent")
    residual_risk: RiskLevel = Field(..., description="Risque résiduel")
    
    # Configuration de monitoring
    automated_check: bool = Field(True, description="Vérification automatisée")
    check_frequency: str = Field("1h", description="Fréquence de vérification")
    check_query: Optional[str] = Field(None, description="Requête de vérification")
    
    # Seuils et métriques
    compliance_threshold: float = Field(95.0, description="Seuil de conformité (%)")
    warning_threshold: float = Field(90.0, description="Seuil d'avertissement (%)")
    
    # Statut
    current_status: ComplianceStatus = Field(..., description="Statut actuel")
    last_assessment_date: datetime = Field(..., description="Dernière évaluation")
    next_assessment_date: datetime = Field(..., description="Prochaine évaluation")
    
    # Responsabilités
    owner: str = Field(..., description="Propriétaire du contrôle")
    reviewer: str = Field(..., description="Réviseur")
    approver: str = Field(..., description="Approbateur")
    
    # Documentation
    documentation_url: Optional[str] = Field(None, description="URL documentation")
    policy_reference: Optional[str] = Field(None, description="Référence politique")
    procedure_reference: Optional[str] = Field(None, description="Référence procédure")
    
    # Evidence et preuve
    evidence_required: List[str] = Field(default_factory=list, description="Preuves requises")
    evidence_collected: List[str] = Field(default_factory=list, description="Preuves collectées")
    
    # Métadonnées
    tags: List[str] = Field(default_factory=list, description="Tags")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('compliance_threshold', 'warning_threshold')
    def validate_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Threshold must be between 0 and 100')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "control_id": "soc2_cc6_1",
                "framework": "soc2",
                "control_number": "CC6.1",
                "title": "Logical and Physical Access Controls",
                "description": "The entity implements logical and physical access controls",
                "category": "Access Control",
                "criticality": "high",
                "automated_check": True,
                "check_frequency": "1h"
            }
        }


class ComplianceViolation(BaseModel):
    """Violation de conformité détectée"""
    
    # Identifiants
    violation_id: str = Field(default_factory=lambda: str(uuid4()), description="ID unique")
    control_id: str = Field(..., description="ID du contrôle violé")
    
    # Classification
    violation_type: ViolationType = Field(..., description="Type de violation")
    severity: RiskLevel = Field(..., description="Sévérité")
    
    # Description
    title: str = Field(..., description="Titre de la violation")
    description: str = Field(..., description="Description détaillée")
    impact_description: str = Field(..., description="Description de l'impact")
    
    # Détection
    detected_at: datetime = Field(default_factory=datetime.utcnow, description="Date détection")
    detection_method: str = Field(..., description="Méthode de détection")
    detection_source: str = Field(..., description="Source de détection")
    
    # Contexte
    affected_systems: List[str] = Field(default_factory=list, description="Systèmes affectés")
    affected_data_types: List[str] = Field(default_factory=list, description="Types données affectées")
    affected_users: int = Field(0, description="Nombre utilisateurs affectés")
    
    # Données techniques
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Données brutes")
    evidence: List[str] = Field(default_factory=list, description="Preuves")
    
    # Remédiation
    remediation_status: RemediationStatus = Field(RemediationStatus.OPEN, description="Statut remédiation")
    remediation_plan: Optional[str] = Field(None, description="Plan de remédiation")
    remediation_deadline: Optional[datetime] = Field(None, description="Échéance remédiation")
    remediation_owner: Optional[str] = Field(None, description="Responsable remédiation")
    
    # Suivi
    acknowledged_at: Optional[datetime] = Field(None, description="Date accusé réception")
    acknowledged_by: Optional[str] = Field(None, description="Accusé réception par")
    resolved_at: Optional[datetime] = Field(None, description="Date résolution")
    resolved_by: Optional[str] = Field(None, description="Résolu par")
    
    # Métadonnées
    tags: List[str] = Field(default_factory=list, description="Tags")
    
    class Config:
        schema_extra = {
            "example": {
                "control_id": "soc2_cc6_1",
                "violation_type": "unauthorized_access",
                "severity": "high",
                "title": "Unauthorized API Access Detected",
                "description": "Multiple failed authentication attempts detected",
                "detection_method": "automated_monitoring",
                "affected_systems": ["api-gateway", "user-service"]
            }
        }


class ComplianceReport(BaseModel):
    """Rapport de conformité"""
    
    # Identifiants
    report_id: str = Field(default_factory=lambda: str(uuid4()), description="ID du rapport")
    framework: ComplianceFramework = Field(..., description="Framework")
    
    # Période
    reporting_period_start: datetime = Field(..., description="Début période")
    reporting_period_end: datetime = Field(..., description="Fin période")
    
    # Scores globaux
    overall_compliance_score: float = Field(..., description="Score conformité global")
    total_controls: int = Field(..., description="Nombre total contrôles")
    compliant_controls: int = Field(..., description="Contrôles conformes")
    non_compliant_controls: int = Field(..., description="Contrôles non conformes")
    
    # Détails par catégorie
    category_scores: Dict[str, float] = Field(default_factory=dict, description="Scores par catégorie")
    
    # Violations
    total_violations: int = Field(0, description="Total violations")
    critical_violations: int = Field(0, description="Violations critiques")
    high_violations: int = Field(0, description="Violations élevées")
    resolved_violations: int = Field(0, description="Violations résolues")
    
    # Tendances
    score_trend: str = Field("stable", description="Tendance score (improving/declining/stable)")
    previous_score: Optional[float] = Field(None, description="Score précédent")
    
    # Recommandations
    recommendations: List[str] = Field(default_factory=list, description="Recommandations")
    priority_actions: List[str] = Field(default_factory=list, description="Actions prioritaires")
    
    # Métadonnées
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: str = Field("automated_system", description="Généré par")
    
    @validator('overall_compliance_score')
    def validate_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Score must be between 0 and 100')
        return v


class ComplianceAuditEntry(BaseModel):
    """Entrée d'audit de conformité"""
    
    # Identifiants
    audit_id: str = Field(default_factory=lambda: str(uuid4()), description="ID audit")
    
    # Événement
    event_type: str = Field(..., description="Type d'événement")
    event_description: str = Field(..., description="Description événement")
    
    # Contexte
    user_id: Optional[str] = Field(None, description="ID utilisateur")
    system_id: str = Field(..., description="ID système")
    resource_id: Optional[str] = Field(None, description="ID ressource")
    
    # Compliance context
    framework: Optional[ComplianceFramework] = Field(None, description="Framework concerné")
    control_id: Optional[str] = Field(None, description="Contrôle concerné")
    
    # Détails techniques
    ip_address: Optional[str] = Field(None, description="Adresse IP")
    user_agent: Optional[str] = Field(None, description="User agent")
    session_id: Optional[str] = Field(None, description="ID session")
    
    # Données
    before_value: Optional[str] = Field(None, description="Valeur avant")
    after_value: Optional[str] = Field(None, description="Valeur après")
    
    # Métadonnées
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Classification
    severity: RiskLevel = Field(RiskLevel.LOW, description="Sévérité")
    sensitive_data: bool = Field(False, description="Données sensibles")
    
    class Config:
        schema_extra = {
            "example": {
                "event_type": "data_access",
                "event_description": "User accessed customer PII data",
                "user_id": "user123",
                "system_id": "customer-db",
                "framework": "gdpr",
                "severity": "medium",
                "sensitive_data": True
            }
        }


class ComplianceMetrics(BaseModel):
    """Métriques de conformité en temps réel"""
    
    # Identifiants
    framework: ComplianceFramework = Field(..., description="Framework")
    tenant_id: Optional[str] = Field(None, description="ID tenant")
    
    # Scores actuels
    current_score: float = Field(..., description="Score actuel")
    target_score: float = Field(95.0, description="Score cible")
    
    # Contrôles
    total_controls: int = Field(..., description="Total contrôles")
    passing_controls: int = Field(..., description="Contrôles passants")
    failing_controls: int = Field(..., description="Contrôles échouants")
    
    # Violations actives
    active_violations: int = Field(0, description="Violations actives")
    critical_violations: int = Field(0, description="Violations critiques")
    overdue_remediations: int = Field(0, description="Remédiations en retard")
    
    # Tendances (24h)
    score_change_24h: float = Field(0.0, description="Changement score 24h")
    new_violations_24h: int = Field(0, description="Nouvelles violations 24h")
    resolved_violations_24h: int = Field(0, description="Violations résolues 24h")
    
    # Audit
    audit_events_count: int = Field(0, description="Nombre événements audit")
    sensitive_operations_count: int = Field(0, description="Opérations sensibles")
    
    # Risques
    overall_risk_score: float = Field(0.0, description="Score risque global")
    high_risk_areas: List[str] = Field(default_factory=list, description="Zones haut risque")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('current_score', 'target_score', 'overall_risk_score')
    def validate_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Score must be between 0 and 100')
        return v


class ComplianceMonitoringService(BaseModel):
    """Service de monitoring de conformité ultra-avancé"""
    
    # Configuration
    service_name: str = Field("compliance-monitoring", description="Nom du service")
    version: str = Field("1.0.0", description="Version")
    
    # Frameworks actifs
    active_frameworks: List[ComplianceFramework] = Field(
        default_factory=list, description="Frameworks actifs"
    )
    
    # Contrôles
    controls: Dict[str, ComplianceControl] = Field(
        default_factory=dict, description="Contrôles configurés"
    )
    
    # Configuration monitoring
    assessment_frequency: str = Field("1h", description="Fréquence évaluation")
    reporting_frequency: str = Field("daily", description="Fréquence rapports")
    
    # Alertes et notifications
    alert_on_violations: bool = Field(True, description="Alerter sur violations")
    notification_channels: List[str] = Field(
        default_factory=list, description="Canaux notification"
    )
    
    # Audit
    audit_enabled: bool = Field(True, description="Audit activé")
    audit_retention_days: int = Field(2555, description="Rétention audit (7 ans)")
    
    # ML et IA
    ml_risk_prediction: bool = Field(True, description="Prédiction risque ML")
    anomaly_detection: bool = Field(True, description="Détection anomalies")
    
    def add_control(self, control: ComplianceControl) -> None:
        """Ajouter un contrôle de conformité"""
        if control.control_id in self.controls:
            raise ValueError(f"Control {control.control_id} already exists")
        self.controls[control.control_id] = control
    
    def get_control(self, control_id: str) -> Optional[ComplianceControl]:
        """Récupérer un contrôle"""
        return self.controls.get(control_id)
    
    def get_controls_by_framework(self, framework: ComplianceFramework) -> List[ComplianceControl]:
        """Récupérer contrôles par framework"""
        return [c for c in self.controls.values() if c.framework == framework]
    
    def calculate_framework_score(self, framework: ComplianceFramework) -> float:
        """Calculer score de conformité pour un framework"""
        controls = self.get_controls_by_framework(framework)
        if not controls:
            return 0.0
        
        compliant_count = sum(
            1 for c in controls 
            if c.current_status == ComplianceStatus.COMPLIANT
        )
        
        return (compliant_count / len(controls)) * 100


# Contrôles prédéfinis pour frameworks majeurs
GDPR_CORE_CONTROLS = [
    ComplianceControl(
        control_id="gdpr_art_6_lawful_basis",
        framework=ComplianceFramework.GDPR,
        control_number="Art. 6",
        title="Lawful Basis for Processing",
        description="Processing of personal data must have a lawful basis",
        objective="Ensure all data processing has valid legal basis",
        category="Legal Basis",
        criticality=RiskLevel.CRITICAL,
        inherent_risk=RiskLevel.HIGH,
        residual_risk=RiskLevel.MEDIUM,
        current_status=ComplianceStatus.COMPLIANT,
        last_assessment_date=datetime.utcnow(),
        next_assessment_date=datetime.utcnow() + timedelta(days=30),
        owner="Data Protection Officer",
        reviewer="Legal Team",
        approver="Chief Privacy Officer"
    ),
    
    ComplianceControl(
        control_id="gdpr_art_25_privacy_by_design",
        framework=ComplianceFramework.GDPR,
        control_number="Art. 25",
        title="Privacy by Design and Default",
        description="Implement privacy by design and default principles",
        objective="Ensure privacy protection is built into systems",
        category="Privacy Engineering",
        criticality=RiskLevel.HIGH,
        inherent_risk=RiskLevel.HIGH,
        residual_risk=RiskLevel.LOW,
        current_status=ComplianceStatus.COMPLIANT,
        last_assessment_date=datetime.utcnow(),
        next_assessment_date=datetime.utcnow() + timedelta(days=90),
        owner="Engineering Team",
        reviewer="Data Protection Officer",
        approver="CTO"
    )
]

SOC2_CORE_CONTROLS = [
    ComplianceControl(
        control_id="soc2_cc6_1_logical_access",
        framework=ComplianceFramework.SOC2,
        control_number="CC6.1",
        title="Logical and Physical Access Controls",
        description="Entity implements logical and physical access controls",
        objective="Protect information and technology assets",
        category="Access Control",
        criticality=RiskLevel.CRITICAL,
        inherent_risk=RiskLevel.HIGH,
        residual_risk=RiskLevel.LOW,
        automated_check=True,
        check_frequency="1h",
        check_query="spotify_unauthorized_access_attempts > 0",
        current_status=ComplianceStatus.COMPLIANT,
        last_assessment_date=datetime.utcnow(),
        next_assessment_date=datetime.utcnow() + timedelta(days=30),
        owner="Security Team",
        reviewer="Internal Audit",
        approver="CISO"
    ),
    
    ComplianceControl(
        control_id="soc2_cc7_1_system_monitoring",
        framework=ComplianceFramework.SOC2,
        control_number="CC7.1",
        title="System Monitoring",
        description="Monitor system components and operations",
        objective="Detect and respond to system anomalies",
        category="System Operations",
        criticality=RiskLevel.HIGH,
        inherent_risk=RiskLevel.MEDIUM,
        residual_risk=RiskLevel.LOW,
        automated_check=True,
        check_frequency="5m",
        current_status=ComplianceStatus.COMPLIANT,
        last_assessment_date=datetime.utcnow(),
        next_assessment_date=datetime.utcnow() + timedelta(days=7),
        owner="Operations Team",
        reviewer="Security Team",
        approver="CTO"
    )
]


def create_default_compliance_service() -> ComplianceMonitoringService:
    """Créer service de conformité avec contrôles par défaut"""
    service = ComplianceMonitoringService(
        active_frameworks=[
            ComplianceFramework.GDPR,
            ComplianceFramework.SOC2,
            ComplianceFramework.ISO27001
        ],
        notification_channels=["slack", "email", "pagerduty"]
    )
    
    # Ajouter contrôles GDPR
    for control in GDPR_CORE_CONTROLS:
        service.add_control(control)
    
    # Ajouter contrôles SOC2  
    for control in SOC2_CORE_CONTROLS:
        service.add_control(control)
    
    return service


# Export des classes principales
__all__ = [
    "ComplianceFramework",
    "ComplianceStatus",
    "RiskLevel",
    "ViolationType",
    "RemediationStatus",
    "ComplianceControl",
    "ComplianceViolation",
    "ComplianceReport",
    "ComplianceAuditEntry",
    "ComplianceMetrics",
    "ComplianceMonitoringService",
    "GDPR_CORE_CONTROLS",
    "SOC2_CORE_CONTROLS",
    "create_default_compliance_service"
]
