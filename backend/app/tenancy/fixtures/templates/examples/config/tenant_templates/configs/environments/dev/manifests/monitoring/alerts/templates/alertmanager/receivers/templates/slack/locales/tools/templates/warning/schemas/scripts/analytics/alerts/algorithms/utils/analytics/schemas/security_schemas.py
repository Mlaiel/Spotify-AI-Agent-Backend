"""
Security Schemas - Ultra-Advanced Edition
========================================

Schémas ultra-avancés pour la sécurité avec audit, compliance, threat detection
et analyse comportementale.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Set, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, PositiveInt, NonNegativeFloat, PositiveFloat, EmailStr


class SecurityEventType(str, Enum):
    """Types d'événements de sécurité."""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PERMISSION_CHANGE = "permission_change"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    THREAT_DETECTED = "threat_detected"


class ThreatLevel(str, Enum):
    """Niveaux de menace."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(BaseModel):
    """Événement de sécurité avec analyse comportementale."""
    
    event_id: UUID4 = Field(default_factory=lambda: UUID4())
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Classification
    event_type: SecurityEventType = Field(..., description="Type d'événement")
    threat_level: ThreatLevel = Field(default=ThreatLevel.LOW, description="Niveau de menace")
    
    # Acteur
    user_id: Optional[UUID4] = Field(None, description="ID utilisateur")
    ip_address: str = Field(..., description="Adresse IP")
    user_agent: Optional[str] = Field(None, description="User agent")
    
    # Contexte
    resource_accessed: Optional[str] = Field(None, description="Ressource accédée")
    action_performed: str = Field(..., description="Action effectuée")
    success: bool = Field(..., description="Succès de l'action")
    
    # Analyse comportementale
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Score d'anomalie")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Score de risque")
    confidence_level: float = Field(default=1.0, ge=0.0, le=1.0, description="Niveau de confiance")
    
    # Géolocalisation
    country: Optional[str] = Field(None, description="Pays")
    city: Optional[str] = Field(None, description="Ville")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Coordonnées")
    
    # Détails techniques
    session_id: Optional[str] = Field(None, description="ID de session")
    device_fingerprint: Optional[str] = Field(None, description="Empreinte device")
    request_headers: Dict[str, str] = Field(default_factory=dict, description="Headers de requête")
    
    # Réponse
    blocked: bool = Field(default=False, description="Action bloquée")
    response_action: Optional[str] = Field(None, description="Action de réponse")
    alert_triggered: bool = Field(default=False, description="Alerte déclenchée")


class AuditLog(BaseModel):
    """Log d'audit complet avec traçabilité."""
    
    log_id: UUID4 = Field(default_factory=lambda: UUID4())
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Acteur
    user_id: UUID4 = Field(..., description="ID utilisateur")
    user_email: EmailStr = Field(..., description="Email utilisateur")
    role: str = Field(..., description="Rôle utilisateur")
    
    # Action
    action: str = Field(..., description="Action effectuée")
    resource_type: str = Field(..., description="Type de ressource")
    resource_id: Optional[str] = Field(None, description="ID de la ressource")
    
    # Détails
    before_state: Optional[Dict[str, Any]] = Field(None, description="État avant")
    after_state: Optional[Dict[str, Any]] = Field(None, description="État après")
    changes: Dict[str, Any] = Field(default_factory=dict, description="Changements")
    
    # Métadonnées
    ip_address: str = Field(..., description="Adresse IP")
    user_agent: Optional[str] = Field(None, description="User agent")
    session_id: Optional[str] = Field(None, description="ID de session")
    
    # Classification
    category: str = Field(..., description="Catégorie d'audit")
    severity: str = Field(default="info", description="Sévérité")
    compliance_relevant: bool = Field(default=True, description="Pertinent pour compliance")
    
    # Rétention
    retention_period_days: int = Field(default=2555, description="Période de rétention")  # 7 ans par défaut
    encrypted: bool = Field(default=True, description="Log chiffré")


class ComplianceReport(BaseModel):
    """Rapport de conformité avec métriques détaillées."""
    
    report_id: UUID4 = Field(default_factory=lambda: UUID4())
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    
    # Période
    report_period_start: datetime = Field(..., description="Début période")
    report_period_end: datetime = Field(..., description="Fin période")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Framework de conformité
    compliance_framework: str = Field(..., description="Framework (GDPR, HIPAA, etc.)")
    version: str = Field(..., description="Version du framework")
    
    # Scores de conformité
    overall_compliance_score: float = Field(..., ge=0.0, le=100.0, description="Score global")
    data_protection_score: float = Field(..., ge=0.0, le=100.0, description="Protection données")
    access_control_score: float = Field(..., ge=0.0, le=100.0, description="Contrôle accès")
    audit_trail_score: float = Field(..., ge=0.0, le=100.0, description="Piste d'audit")
    
    # Violations détectées
    violations: List[Dict[str, Any]] = Field(default_factory=list, description="Violations")
    critical_issues: int = Field(default=0, ge=0, description="Problèmes critiques")
    high_issues: int = Field(default=0, ge=0, description="Problèmes élevés")
    medium_issues: int = Field(default=0, ge=0, description="Problèmes moyens")
    
    # Recommandations
    recommendations: List[str] = Field(default_factory=list, description="Recommandations")
    remediation_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Actions correctives")
    
    # Métadonnées
    generated_by: UUID4 = Field(..., description="Généré par")
    reviewed: bool = Field(default=False, description="Rapport révisé")
    approved: bool = Field(default=False, description="Rapport approuvé")
    approved_by: Optional[UUID4] = Field(None, description="Approuvé par")


# Export
__all__ = [
    "SecurityEventType",
    "ThreatLevel", 
    "SecurityEvent",
    "AuditLog",
    "ComplianceReport"
]
