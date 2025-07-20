"""
Advanced Security Monitoring - Industrial Grade Cybersecurity Observability
==========================================================================

Ce module fournit une architecture de monitoring de sécurité ultra-avancée
pour détection de menaces, analyse comportementale et réponse aux incidents.

Features:
- Real-time threat detection and analysis
- Behavioral analytics and anomaly detection  
- SIEM integration and correlation
- Zero-trust security monitoring
- Compliance and regulatory monitoring
- Incident response automation
- Threat intelligence integration
- Advanced persistent threat (APT) detection
"""

from typing import Dict, List, Optional, Union, Any, Set
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import json
from uuid import uuid4
from ipaddress import IPv4Address, IPv6Address


class ThreatLevel(str, Enum):
    """Niveaux de menace"""
    CRITICAL = "critical"           # Menace critique - action immédiate
    HIGH = "high"                   # Menace élevée - action dans 15 min
    MEDIUM = "medium"               # Menace moyenne - action dans 1h
    LOW = "low"                     # Menace faible - surveillance
    INFO = "info"                   # Information - logging


class ThreatType(str, Enum):
    """Types de menaces"""
    MALWARE = "malware"
    PHISHING = "phishing"
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    DDoS = "ddos"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INSIDER_THREAT = "insider_threat"
    APT = "apt"
    ZERO_DAY = "zero_day"
    RANSOMWARE = "ransomware"
    SOCIAL_ENGINEERING = "social_engineering"
    SUPPLY_CHAIN = "supply_chain"
    IOT_COMPROMISE = "iot_compromise"


class AttackVector(str, Enum):
    """Vecteurs d'attaque"""
    NETWORK = "network"
    EMAIL = "email"
    WEB_APPLICATION = "web_application"
    ENDPOINT = "endpoint"
    CLOUD = "cloud"
    MOBILE = "mobile"
    PHYSICAL = "physical"
    SOCIAL = "social"
    SUPPLY_CHAIN = "supply_chain"
    INSIDER = "insider"


class EventStatus(str, Enum):
    """Statuts d'événement de sécurité"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    CLOSED = "closed"


class IncidentSeverity(str, Enum):
    """Sévérité d'incident"""
    SEV1 = "sev1"                  # Critique - Service indisponible
    SEV2 = "sev2"                  # Majeur - Fonctionnalité compromise
    SEV3 = "sev3"                  # Mineur - Impact limité
    SEV4 = "sev4"                  # Info - Pas d'impact service


class MITREATTACKTactic(str, Enum):
    """Tactiques MITRE ATT&CK"""
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command_and_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


class SecurityEvent(BaseModel):
    """Événement de sécurité détecté"""
    
    # Identifiants
    event_id: str = Field(default_factory=lambda: str(uuid4()), description="ID unique événement")
    correlation_id: Optional[str] = Field(None, description="ID corrélation")
    
    # Classification
    threat_type: ThreatType = Field(..., description="Type de menace")
    threat_level: ThreatLevel = Field(..., description="Niveau de menace")
    attack_vector: AttackVector = Field(..., description="Vecteur d'attaque")
    
    # MITRE ATT&CK mapping
    mitre_tactics: List[MITREATTACKTactic] = Field(default_factory=list, description="Tactiques MITRE")
    mitre_techniques: List[str] = Field(default_factory=list, description="Techniques MITRE")
    
    # Détails événement
    title: str = Field(..., description="Titre événement")
    description: str = Field(..., description="Description détaillée")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Données brutes")
    
    # Source et cible
    source_ip: Optional[str] = Field(None, description="IP source")
    source_country: Optional[str] = Field(None, description="Pays source")
    source_asn: Optional[str] = Field(None, description="ASN source")
    target_ip: Optional[str] = Field(None, description="IP cible")
    target_hostname: Optional[str] = Field(None, description="Hostname cible")
    
    # Utilisateur et identité
    user_id: Optional[str] = Field(None, description="ID utilisateur")
    username: Optional[str] = Field(None, description="Nom utilisateur")
    user_agent: Optional[str] = Field(None, description="User agent")
    session_id: Optional[str] = Field(None, description="ID session")
    
    # Contexte réseau
    port: Optional[int] = Field(None, description="Port")
    protocol: Optional[str] = Field(None, description="Protocole")
    url: Optional[str] = Field(None, description="URL")
    http_method: Optional[str] = Field(None, description="Méthode HTTP")
    http_status: Optional[int] = Field(None, description="Status HTTP")
    
    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp événement")
    first_seen: datetime = Field(default_factory=datetime.utcnow, description="Première occurrence")
    last_seen: datetime = Field(default_factory=datetime.utcnow, description="Dernière occurrence")
    duration: Optional[timedelta] = Field(None, description="Durée")
    
    # Indicateurs
    indicators_of_compromise: List[str] = Field(default_factory=list, description="IoCs")
    hash_values: Dict[str, str] = Field(default_factory=dict, description="Hashes (MD5, SHA1, SHA256)")
    
    # Score et confiance
    confidence_score: float = Field(..., description="Score de confiance")
    risk_score: float = Field(..., description="Score de risque")
    false_positive_probability: float = Field(0.0, description="Probabilité faux positif")
    
    # Impact
    affected_assets: List[str] = Field(default_factory=list, description="Assets affectés")
    affected_users: List[str] = Field(default_factory=list, description="Utilisateurs affectés")
    business_impact: Optional[str] = Field(None, description="Impact business")
    
    # Statut et traitement
    status: EventStatus = Field(EventStatus.DETECTED, description="Statut événement")
    assigned_to: Optional[str] = Field(None, description="Assigné à")
    
    # Threat Intelligence
    threat_feed_matches: List[str] = Field(default_factory=list, description="Correspondances threat feeds")
    reputation_score: Optional[float] = Field(None, description="Score réputation")
    
    # Géolocalisation
    geolocation: Dict[str, Any] = Field(default_factory=dict, description="Géolocalisation")
    
    # Tags et labels
    tags: List[str] = Field(default_factory=list, description="Tags")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels")
    
    @validator('confidence_score', 'risk_score', 'false_positive_probability')
    def validate_scores(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v
    
    @validator('port')
    def validate_port(cls, v):
        if v is not None and not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "threat_type": "brute_force",
                "threat_level": "high",
                "attack_vector": "network",
                "title": "Brute Force Attack Detected",
                "description": "Multiple failed login attempts from suspicious IP",
                "source_ip": "192.168.1.100",
                "username": "admin",
                "confidence_score": 0.95,
                "risk_score": 0.8,
                "mitre_tactics": ["credential_access"]
            }
        }


class BehavioralAnomaly(BaseModel):
    """Anomalie comportementale détectée"""
    
    # Identifiants
    anomaly_id: str = Field(default_factory=lambda: str(uuid4()), description="ID anomalie")
    user_id: Optional[str] = Field(None, description="ID utilisateur")
    entity_id: str = Field(..., description="ID entité (user, asset, etc.)")
    entity_type: str = Field(..., description="Type entité")
    
    # Détection
    anomaly_type: str = Field(..., description="Type d'anomalie")
    severity: ThreatLevel = Field(..., description="Sévérité")
    
    # Description
    title: str = Field(..., description="Titre anomalie")
    description: str = Field(..., description="Description")
    
    # Comportement
    baseline_behavior: Dict[str, Any] = Field(default_factory=dict, description="Comportement baseline")
    observed_behavior: Dict[str, Any] = Field(default_factory=dict, description="Comportement observé")
    deviation_score: float = Field(..., description="Score de déviation")
    
    # Contexte temporel
    detection_window_start: datetime = Field(..., description="Début fenêtre détection")
    detection_window_end: datetime = Field(..., description="Fin fenêtre détection")
    baseline_period: str = Field(..., description="Période baseline")
    
    # ML et algorithmes
    detection_algorithm: str = Field(..., description="Algorithme de détection")
    model_version: str = Field(..., description="Version modèle")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Importance features")
    
    # Actions recommandées
    recommended_actions: List[str] = Field(default_factory=list, description="Actions recommandées")
    risk_mitigation: List[str] = Field(default_factory=list, description="Mitigation des risques")
    
    # Statut
    investigated: bool = Field(False, description="Investigué")
    confirmed_malicious: Optional[bool] = Field(None, description="Confirmé malicieux")
    
    # Timestamp
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('deviation_score')
    def validate_deviation_score(cls, v):
        if v < 0:
            raise ValueError('Deviation score must be positive')
        return v


class SecurityIncident(BaseModel):
    """Incident de sécurité"""
    
    # Identifiants
    incident_id: str = Field(default_factory=lambda: str(uuid4()), description="ID incident")
    title: str = Field(..., description="Titre incident")
    
    # Classification
    incident_type: ThreatType = Field(..., description="Type incident")
    severity: IncidentSeverity = Field(..., description="Sévérité")
    
    # Description
    description: str = Field(..., description="Description détaillée")
    summary: str = Field(..., description="Résumé exécutif")
    
    # Événements liés
    related_events: List[str] = Field(default_factory=list, description="IDs événements liés")
    related_anomalies: List[str] = Field(default_factory=list, description="IDs anomalies liées")
    
    # Timeline
    detected_at: datetime = Field(..., description="Détection")
    reported_at: datetime = Field(..., description="Signalement")
    investigation_started_at: Optional[datetime] = Field(None, description="Début investigation")
    contained_at: Optional[datetime] = Field(None, description="Confinement")
    resolved_at: Optional[datetime] = Field(None, description="Résolution")
    
    # Impact
    affected_systems: List[str] = Field(default_factory=list, description="Systèmes affectés")
    affected_data: List[str] = Field(default_factory=list, description="Données affectées")
    affected_users_count: int = Field(0, description="Nombre utilisateurs affectés")
    business_impact_description: str = Field("", description="Description impact business")
    estimated_cost: Optional[float] = Field(None, description="Coût estimé")
    
    # Réponse
    response_team: List[str] = Field(default_factory=list, description="Équipe de réponse")
    incident_commander: Optional[str] = Field(None, description="Commandant incident")
    actions_taken: List[str] = Field(default_factory=list, description="Actions prises")
    
    # Communication
    stakeholders_notified: List[str] = Field(default_factory=list, description="Parties prenantes notifiées")
    public_disclosure: bool = Field(False, description="Divulgation publique")
    regulatory_notification: bool = Field(False, description="Notification réglementaire")
    
    # Lessons learned
    root_cause: Optional[str] = Field(None, description="Cause racine")
    lessons_learned: List[str] = Field(default_factory=list, description="Leçons apprises")
    preventive_measures: List[str] = Field(default_factory=list, description="Mesures préventives")
    
    # Métadonnées
    created_by: str = Field(..., description="Créé par")
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list, description="Tags")


class ThreatIntelligence(BaseModel):
    """Intelligence de menace"""
    
    # Identifiants
    intel_id: str = Field(default_factory=lambda: str(uuid4()), description="ID intel")
    source: str = Field(..., description="Source intel")
    
    # Classification
    threat_type: ThreatType = Field(..., description="Type menace")
    threat_actor: Optional[str] = Field(None, description="Acteur menace")
    campaign: Optional[str] = Field(None, description="Campagne")
    
    # IoCs (Indicators of Compromise)
    ip_addresses: List[str] = Field(default_factory=list, description="Adresses IP")
    domains: List[str] = Field(default_factory=list, description="Domaines")
    urls: List[str] = Field(default_factory=list, description="URLs")
    file_hashes: Dict[str, str] = Field(default_factory=dict, description="Hashes fichiers")
    email_addresses: List[str] = Field(default_factory=list, description="Adresses email")
    
    # TTPs (Tactics, Techniques, Procedures)
    mitre_tactics: List[MITREATTACKTactic] = Field(default_factory=list, description="Tactiques")
    mitre_techniques: List[str] = Field(default_factory=list, description="Techniques")
    
    # Confiance et fiabilité
    confidence_level: str = Field(..., description="Niveau confiance")
    reliability_score: float = Field(..., description="Score fiabilité")
    
    # Validité
    first_seen: datetime = Field(..., description="Première observation")
    last_seen: datetime = Field(..., description="Dernière observation")
    expires_at: Optional[datetime] = Field(None, description="Expiration")
    
    # Description
    description: str = Field(..., description="Description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contexte")
    
    # Tags
    tags: List[str] = Field(default_factory=list, description="Tags")
    
    @validator('reliability_score')
    def validate_reliability(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Reliability score must be between 0 and 1')
        return v


class SecurityMetrics(BaseModel):
    """Métriques de sécurité en temps réel"""
    
    # Identifiants
    tenant_id: Optional[str] = Field(None, description="ID tenant")
    
    # Événements
    total_events_24h: int = Field(0, description="Total événements 24h")
    critical_events_24h: int = Field(0, description="Événements critiques 24h")
    high_events_24h: int = Field(0, description="Événements élevés 24h")
    
    # Incidents
    active_incidents: int = Field(0, description="Incidents actifs")
    incidents_resolved_24h: int = Field(0, description="Incidents résolus 24h")
    mean_time_to_detect: float = Field(0.0, description="MTTD (minutes)")
    mean_time_to_respond: float = Field(0.0, description="MTTR (minutes)")
    
    # Anomalies comportementales
    behavioral_anomalies_24h: int = Field(0, description="Anomalies comportementales 24h")
    confirmed_malicious_rate: float = Field(0.0, description="Taux confirmé malicieux")
    
    # Authentification
    failed_logins_24h: int = Field(0, description="Échecs connexion 24h")
    suspicious_logins_24h: int = Field(0, description="Connexions suspectes 24h")
    mfa_bypass_attempts: int = Field(0, description="Tentatives contournement MFA")
    
    # Réseau
    blocked_ips_24h: int = Field(0, description="IPs bloquées 24h")
    malicious_domains_blocked: int = Field(0, description="Domaines malicieux bloqués")
    data_exfiltration_attempts: int = Field(0, description="Tentatives exfiltration")
    
    # Vulnérabilités
    critical_vulnerabilities: int = Field(0, description="Vulnérabilités critiques")
    high_vulnerabilities: int = Field(0, description="Vulnérabilités élevées")
    unpatched_systems: int = Field(0, description="Systèmes non patchés")
    
    # Conformité
    compliance_violations_24h: int = Field(0, description="Violations conformité 24h")
    policy_violations_24h: int = Field(0, description="Violations politique 24h")
    
    # Threat Intelligence
    new_iocs_24h: int = Field(0, description="Nouveaux IoCs 24h")
    threat_intel_matches: int = Field(0, description="Correspondances threat intel")
    
    # Score global
    security_posture_score: float = Field(0.0, description="Score posture sécurité")
    risk_score: float = Field(0.0, description="Score risque global")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('security_posture_score', 'risk_score', 'confirmed_malicious_rate')
    def validate_scores(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Score must be between 0 and 100')
        return v


class SecurityMonitoringService(BaseModel):
    """Service de monitoring de sécurité ultra-avancé"""
    
    # Configuration
    service_name: str = Field("security-monitoring", description="Nom du service")
    version: str = Field("1.0.0", description="Version")
    
    # Configuration détection
    threat_detection_enabled: bool = Field(True, description="Détection menaces activée")
    behavioral_analytics_enabled: bool = Field(True, description="Analytics comportemental activé")
    threat_intelligence_enabled: bool = Field(True, description="Threat intelligence activé")
    
    # SIEM Integration
    siem_integration: bool = Field(True, description="Intégration SIEM")
    log_correlation_enabled: bool = Field(True, description="Corrélation logs activée")
    
    # ML et IA
    ml_anomaly_detection: bool = Field(True, description="Détection anomalies ML")
    auto_investigation: bool = Field(False, description="Investigation automatique")
    
    # Réponse automatisée
    auto_response_enabled: bool = Field(True, description="Réponse automatique activée")
    auto_blocking_enabled: bool = Field(True, description="Blocage automatique activé")
    
    # Alertes et notifications
    alert_channels: List[str] = Field(default_factory=list, description="Canaux d'alerte")
    incident_escalation: bool = Field(True, description="Escalade incidents")
    
    # Threat Intelligence
    threat_feeds: List[str] = Field(default_factory=list, description="Feeds de menaces")
    ioc_enrichment: bool = Field(True, description="Enrichissement IoCs")
    
    # Compliance
    compliance_monitoring: bool = Field(True, description="Monitoring conformité")
    audit_logging: bool = Field(True, description="Logs d'audit")
    
    # Retention
    event_retention_days: int = Field(365, description="Rétention événements")
    incident_retention_days: int = Field(2555, description="Rétention incidents (7 ans)")


# Événements prédéfinis pour Spotify AI Agent
SPOTIFY_SECURITY_EVENTS = [
    {
        "threat_type": ThreatType.BRUTE_FORCE,
        "threat_level": ThreatLevel.HIGH,
        "attack_vector": AttackVector.WEB_APPLICATION,
        "title": "Brute Force Attack on Spotify API",
        "description": "Multiple failed authentication attempts detected on Spotify API endpoints",
        "mitre_tactics": [MITREATTACKTactic.CREDENTIAL_ACCESS],
        "mitre_techniques": ["T1110.001"],
        "confidence_score": 0.95,
        "risk_score": 0.8
    },
    {
        "threat_type": ThreatType.DATA_EXFILTRATION,
        "threat_level": ThreatLevel.CRITICAL,
        "attack_vector": AttackVector.INSIDER,
        "title": "Potential Data Exfiltration Detected",
        "description": "Unusual data access patterns detected for user music preferences",
        "mitre_tactics": [MITREATTACKTactic.COLLECTION, MITREATTACKTactic.EXFILTRATION],
        "confidence_score": 0.85,
        "risk_score": 0.9
    }
]


def create_default_security_monitoring_service() -> SecurityMonitoringService:
    """Créer service de monitoring sécurité par défaut"""
    return SecurityMonitoringService(
        alert_channels=["slack", "pagerduty", "email", "sms"],
        threat_feeds=["misp", "threatfox", "abuse.ch", "virustotal"],
        siem_integration=True,
        ml_anomaly_detection=True,
        auto_response_enabled=True,
        compliance_monitoring=True
    )


# Export des classes principales
__all__ = [
    "ThreatLevel",
    "ThreatType",
    "AttackVector",
    "EventStatus", 
    "IncidentSeverity",
    "MITREATTACKTactic",
    "SecurityEvent",
    "BehavioralAnomaly",
    "SecurityIncident",
    "ThreatIntelligence",
    "SecurityMetrics",
    "SecurityMonitoringService",
    "SPOTIFY_SECURITY_EVENTS",
    "create_default_security_monitoring_service"
]
