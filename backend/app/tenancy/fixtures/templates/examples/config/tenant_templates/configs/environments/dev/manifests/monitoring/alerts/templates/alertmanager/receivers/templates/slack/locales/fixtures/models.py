"""
Modèles de Données Enterprise pour Système de Fixtures Multi-Tenant
==================================================================

Ce module définit tous les modèles de données avancés pour le système de fixtures
d'alertes multi-tenant avec support complet de l'IA, compliance, sécurité et performance.

🎯 Architecture: Domain Driven Design + Event Sourcing + CQRS
🔒 Sécurité: Chiffrement natif + Validation stricte + Audit complet  
⚡ Performance: Optimisation mémoire + Sérialisation rapide + Cache intelligent
🧠 IA: ML intégré + Prédictions + Anomaly Detection + Auto-optimization

Auteur: Fahed Mlaiel - Lead Developer & AI Architect
Équipe: DevOps/ML/Security/Backend Experts
Version: 3.0.0-enterprise
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
import hashlib
from pathlib import Path
from decimal import Decimal
import re

# Imports avancés pour la stack enterprise
from pydantic import BaseModel, Field, validator, root_validator
from sqlalchemy import Column, String, DateTime, Integer, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import numpy as np
from cryptography.fernet import Fernet

Base = declarative_base()

# ============================================================================================
# ENUMS ET TYPES DE BASE
# ============================================================================================

class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes avec priorités numériques."""
    CRITICAL = "critical"    # P0 - Résolution immédiate requise
    HIGH = "high"           # P1 - Résolution sous 1h
    MEDIUM = "medium"       # P2 - Résolution sous 4h
    LOW = "low"            # P3 - Résolution sous 24h
    INFO = "info"          # P4 - Informatif
    DEBUG = "debug"        # P5 - Debug uniquement

class Environment(str, Enum):
    """Environnements de déploiement supportés."""
    DEV = "dev"
    TEST = "test"
    STAGING = "staging"
    PREPROD = "preprod"
    PROD = "prod"
    DISASTER_RECOVERY = "dr"

class Locale(str, Enum):
    """Locales supportées avec codes ISO 639-1."""
    FR = "fr"  # Français
    EN = "en"  # English
    DE = "de"  # Deutsch
    ES = "es"  # Español
    IT = "it"  # Italiano
    PT = "pt"  # Português
    RU = "ru"  # Русский
    ZH = "zh"  # 中文
    JA = "ja"  # 日本語
    KO = "ko"  # 한국어
    AR = "ar"  # العربية
    HI = "hi"  # हिन्दी

class NotificationChannel(str, Enum):
    """Canaux de notification supportés."""
    SLACK = "slack"
    TEAMS = "teams"
    DISCORD = "discord"
    EMAIL = "email"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    PUSH = "push"
    VOICE = "voice"
    TELEGRAM = "telegram"

class AlertType(str, Enum):
    """Types d'alertes métier."""
    SYSTEM = "system"
    APPLICATION = "application"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    INFRASTRUCTURE = "infrastructure"
    COMPLIANCE = "compliance"
    ML_MODEL = "ml_model"
    USER_BEHAVIOR = "user_behavior"
    FINANCIAL = "financial"

class SecurityLevel(IntEnum):
    """Niveaux de sécurité avec priorités."""
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3
    TOP_SECRET = 4

class ComplianceStandard(str, Enum):
    """Standards de compliance supportés."""
    GDPR = "gdpr"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOX = "sox"
    NIST = "nist"

# ============================================================================================
# MODÈLES DE BASE AVEC MIXINS
# ============================================================================================

@dataclass
class BaseEntity:
    """Entité de base avec métadonnées communes."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = field(default="1.0.0")
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise en dictionnaire."""
        data = asdict(self)
        data['tags'] = list(self.tags)
        return data
    
    def get_hash(self) -> str:
        """Génère un hash unique de l'entité."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

@dataclass
class AuditableMixin:
    """Mixin pour l'auditabilité."""
    created_by: str = field(default="system")
    updated_by: str = field(default="system")
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_audit_entry(self, action: str, user: str, details: Dict[str, Any] = None):
        """Ajoute une entrée d'audit."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user": user,
            "details": details or {}
        }
        self.audit_trail.append(entry)

@dataclass
class EncryptableMixin:
    """Mixin pour le chiffrement des données sensibles."""
    encryption_key_id: Optional[str] = None
    encrypted_fields: Set[str] = field(default_factory=set)
    
    def encrypt_field(self, field_name: str, cipher: Fernet) -> None:
        """Chiffre un champ spécifique."""
        if hasattr(self, field_name):
            value = getattr(self, field_name)
            if value and isinstance(value, str):
                encrypted_value = cipher.encrypt(value.encode()).decode()
                setattr(self, field_name, encrypted_value)
                self.encrypted_fields.add(field_name)

# ============================================================================================
# MODÈLES PRINCIPAUX D'ALERTES
# ============================================================================================

@dataclass
class SlackAlert(BaseEntity, AuditableMixin, EncryptableMixin):
    """Modèle complet d'une alerte Slack avec toutes les métadonnées."""
    tenant_id: str
    environment: Environment
    severity: AlertSeverity
    alert_type: AlertType
    title: str
    message: str
    channel: str
    locale: Locale = Locale.FR
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    
    # Contexte et enrichissement
    source_system: str = "achiri-ai-agent"
    correlation_id: Optional[str] = None
    incident_id: Optional[str] = None
    parent_alert_id: Optional[str] = None
    
    # Données techniques
    raw_data: Dict[str, Any] = field(default_factory=dict)
    enriched_data: Dict[str, Any] = field(default_factory=dict)
    ml_predictions: Dict[str, Any] = field(default_factory=dict)
    
    # Statut et lifecycle
    status: str = "new"
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # SLA et escalation
    sla_deadline: Optional[datetime] = None
    escalation_level: int = 0
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Métriques et performance
    creation_latency_ms: Optional[float] = None
    delivery_latency_ms: Optional[float] = None
    processing_time_ms: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Vérifie si l'alerte a expiré son SLA."""
        return self.sla_deadline and datetime.utcnow() > self.sla_deadline
    
    def get_priority_score(self) -> int:
        """Calcule un score de priorité basé sur la sévérité et l'âge."""
        severity_scores = {
            AlertSeverity.CRITICAL: 1000,
            AlertSeverity.HIGH: 750,
            AlertSeverity.MEDIUM: 500,
            AlertSeverity.LOW: 250,
            AlertSeverity.INFO: 100,
            AlertSeverity.DEBUG: 50
        }
        
        base_score = severity_scores.get(self.severity, 0)
        age_hours = (datetime.utcnow() - self.created_at).total_seconds() / 3600
        age_penalty = min(age_hours * 10, 500)  # Max 500 points pour l'âge
        
        return int(base_score + age_penalty)

@dataclass
class AlertTemplate(BaseEntity, AuditableMixin):
    """Template d'alerte avec support multi-langue et IA."""
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    locale: Locale
    channel: NotificationChannel
    
    # Contenu du template
    title_template: str
    message_template: str
    rich_content: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration du template
    variables: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    formatting_rules: Dict[str, Any] = field(default_factory=dict)
    
    # IA et optimisation
    ml_optimized: bool = False
    optimization_score: float = 0.0
    usage_statistics: Dict[str, Any] = field(default_factory=dict)
    effectiveness_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Validation et tests
    last_tested: Optional[datetime] = None
    test_results: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    def render(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rend le template avec le contexte donné."""
        # Implémentation basique - sera enrichie par TemplateRenderer
        return {
            "title": self.title_template.format(**context),
            "message": self.message_template.format(**context),
            "rich_content": self.rich_content
        }

# ============================================================================================
# MODÈLES DE CONFIGURATION
# ============================================================================================

@dataclass
class TenantFixture(BaseEntity, AuditableMixin):
    """Configuration de fixture pour un tenant spécifique."""
    tenant_id: str
    environment: Environment
    
    # Configuration d'alerte
    alert_config: Dict[str, Any] = field(default_factory=dict)
    notification_preferences: Dict[str, Any] = field(default_factory=dict)
    escalation_matrix: Dict[str, Any] = field(default_factory=dict)
    
    # Limites et quotas
    daily_alert_limit: int = 1000
    rate_limit_per_minute: int = 10
    storage_quota_mb: int = 100
    
    # Sécurité et compliance
    security_policies: List[str] = field(default_factory=list)
    compliance_requirements: List[ComplianceStandard] = field(default_factory=list)
    data_retention_days: int = 365
    
    # Intégrations
    slack_config: Dict[str, Any] = field(default_factory=dict)
    teams_config: Dict[str, Any] = field(default_factory=dict)
    webhook_endpoints: List[str] = field(default_factory=list)
    
    # ML et analytics
    enable_ml_predictions: bool = True
    enable_anomaly_detection: bool = True
    ml_model_preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityPolicy(BaseEntity, AuditableMixin):
    """Politique de sécurité pour les alertes."""
    name: str
    description: str
    policy_type: str
    
    # Règles de sécurité
    encryption_required: bool = True
    audit_required: bool = True
    approval_required: bool = False
    
    # Contrôles d'accès
    allowed_users: List[str] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=list)
    restricted_channels: List[str] = field(default_factory=list)
    
    # Validation et conformité
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Monitoring
    violations: List[Dict[str, Any]] = field(default_factory=list)
    effectiveness_score: float = 0.0

# ============================================================================================
# MODÈLES DE PERFORMANCE ET MÉTRIQUES
# ============================================================================================

@dataclass
class PerformanceMetric(BaseEntity):
    """Métrique de performance pour le système."""
    metric_name: str
    metric_type: str
    value: float
    unit: str
    
    # Contexte
    tenant_id: Optional[str] = None
    environment: Optional[Environment] = None
    component: Optional[str] = None
    
    # Seuils et alertes
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Agrégation
    aggregation_period: str = "1m"
    aggregation_function: str = "avg"
    
    # Historique
    historical_values: List[Tuple[datetime, float]] = field(default_factory=list)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TenantMetrics(BaseEntity):
    """Métriques agrégées par tenant."""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    
    # Métriques d'usage
    total_alerts: int = 0
    alerts_by_severity: Dict[str, int] = field(default_factory=dict)
    alerts_by_channel: Dict[str, int] = field(default_factory=dict)
    
    # Métriques de performance
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Métriques de qualité
    false_positive_rate: float = 0.0
    alert_resolution_rate: float = 0.0
    avg_time_to_resolution_hours: float = 0.0
    
    # Métriques business
    cost_per_alert: Decimal = Decimal('0.00')
    roi_score: float = 0.0
    satisfaction_score: float = 0.0

# ============================================================================================
# MODÈLES ML ET IA
# ============================================================================================

@dataclass
class MLModel(BaseEntity, AuditableMixin):
    """Modèle de Machine Learning pour les prédictions."""
    name: str
    model_type: str  # "prediction", "classification", "anomaly_detection", etc.
    algorithm: str
    
    # Configuration du modèle
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    target_variable: Optional[str] = None
    
    # Données d'entraînement
    training_data_path: Optional[str] = None
    training_completed_at: Optional[datetime] = None
    validation_score: Optional[float] = None
    
    # Performance
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Déploiement
    deployed: bool = False
    deployment_environment: Optional[Environment] = None
    serving_endpoint: Optional[str] = None
    
    # Monitoring
    prediction_count: int = 0
    avg_prediction_time_ms: float = 0.0
    drift_detected: bool = False
    last_retrain_date: Optional[datetime] = None

@dataclass
class MLPrediction(BaseEntity):
    """Prédiction générée par un modèle ML."""
    model_id: str
    input_data: Dict[str, Any]
    prediction: Dict[str, Any]
    confidence_score: float
    
    # Contexte
    tenant_id: Optional[str] = None
    alert_id: Optional[str] = None
    
    # Validation
    actual_outcome: Optional[Dict[str, Any]] = None
    prediction_accuracy: Optional[float] = None
    
    # Performance
    processing_time_ms: float = 0.0

@dataclass
class AnomalyEvent(BaseEntity):
    """Événement d'anomalie détecté."""
    detection_model: str
    anomaly_type: str
    severity_score: float
    
    # Données de l'anomalie
    anomaly_data: Dict[str, Any]
    normal_baseline: Dict[str, Any]
    deviation_score: float
    
    # Contexte
    tenant_id: Optional[str] = None
    source_system: Optional[str] = None
    
    # Actions
    alert_generated: bool = False
    alert_id: Optional[str] = None
    auto_remediation_attempted: bool = False
    remediation_successful: Optional[bool] = None

# ============================================================================================
# MODÈLES D'ÉVÉNEMENTS ET AUDIT
# ============================================================================================

@dataclass
class AuditEvent(BaseEntity):
    """Événement d'audit système."""
    event_type: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    
    # Détails de l'événement
    event_data: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Sécurité
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    risk_score: float = 0.0
    
    # Compliance
    compliance_relevant: bool = False
    retention_period_days: int = 2555  # 7 ans par défaut

@dataclass
class MonitoringEvent(BaseEntity):
    """Événement de monitoring système."""
    source: str
    event_type: str
    severity: AlertSeverity
    
    # Données de l'événement
    event_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Contexte
    tenant_id: Optional[str] = None
    environment: Optional[Environment] = None
    service_name: Optional[str] = None
    
    # Traitement
    processed: bool = False
    alert_generated: bool = False
    suppressed: bool = False
    suppression_reason: Optional[str] = None

# ============================================================================================
# MODÈLES DE GESTION OPÉRATIONNELLE
# ============================================================================================

@dataclass
class BusinessRule(BaseEntity, AuditableMixin):
    """Règle métier pour l'automatisation."""
    name: str
    description: str
    rule_type: str
    
    # Définition de la règle
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Configuration
    enabled: bool = True
    priority: int = 100
    
    # Exécution
    execution_count: int = 0
    last_execution: Optional[datetime] = None
    success_rate: float = 0.0
    
    # Validation
    dry_run_results: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class SLAPolicy(BaseEntity, AuditableMixin):
    """Politique de SLA pour les alertes."""
    name: str
    description: str
    
    # Définition du SLA
    severity_targets: Dict[AlertSeverity, timedelta] = field(default_factory=dict)
    availability_target: float = 99.9
    response_time_target_ms: float = 1000.0
    
    # Monitoring
    current_compliance: float = 0.0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Actions
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    penalty_actions: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EscalationEvent(BaseEntity):
    """Événement d'escalation d'alerte."""
    alert_id: str
    escalation_level: int
    escalation_reason: str
    
    # Destinataires
    escalated_to: List[str] = field(default_factory=list)
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    
    # Timing
    escalation_delay: timedelta = timedelta(minutes=15)
    max_escalations: int = 3
    
    # Statut
    acknowledged: bool = False
    resolved: bool = False

# ============================================================================================
# MODÈLES DE SANTÉ ET RÉCUPÉRATION
# ============================================================================================

@dataclass
class HealthCheckResult(BaseEntity):
    """Résultat d'un health check système."""
    component: str
    check_type: str
    status: str  # "healthy", "degraded", "unhealthy"
    
    # Détails
    response_time_ms: float = 0.0
    error_message: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    # Seuils
    warning_threshold_ms: float = 1000.0
    critical_threshold_ms: float = 5000.0

@dataclass
class BackupRecord(BaseEntity):
    """Enregistrement de sauvegarde."""
    backup_type: str
    file_path: str
    size_bytes: int
    
    # Métadonnées
    compression_ratio: float = 0.0
    checksum: str = ""
    encryption_enabled: bool = True
    
    # Validation
    integrity_verified: bool = False
    restoration_tested: bool = False
    
    # Rétention
    retention_period_days: int = 90
    expires_at: Optional[datetime] = None

@dataclass
class RecoveryRecord(BaseEntity):
    """Enregistrement de récupération."""
    incident_id: str
    recovery_type: str
    backup_used: Optional[str] = None
    
    # Timing
    recovery_started_at: datetime = field(default_factory=datetime.utcnow)
    recovery_completed_at: Optional[datetime] = None
    total_recovery_time: Optional[timedelta] = None
    
    # Résultats
    success: bool = False
    data_loss_amount: int = 0
    services_restored: List[str] = field(default_factory=list)
    
    # Post-mortem
    root_cause: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    improvement_actions: List[str] = field(default_factory=list)

# ============================================================================================
# MODÈLES PYDANTIC POUR VALIDATION API
# ============================================================================================

class SlackAlertAPI(BaseModel):
    """Modèle API pour création d'alerte Slack."""
    tenant_id: str = Field(..., description="ID du tenant")
    severity: AlertSeverity = Field(..., description="Niveau de sévérité")
    alert_type: AlertType = Field(..., description="Type d'alerte")
    title: str = Field(..., min_length=1, max_length=200, description="Titre de l'alerte")
    message: str = Field(..., min_length=1, max_length=4000, description="Message de l'alerte")
    channel: str = Field(..., regex=r'^#[a-zA-Z0-9_-]+$', description="Canal Slack")
    locale: Locale = Field(default=Locale.FR, description="Locale pour l'internationalisation")
    
    # Champs optionnels
    correlation_id: Optional[str] = Field(None, description="ID de corrélation")
    source_system: str = Field(default="achiri-ai-agent", description="Système source")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Données brutes")
    
    @validator('channel')
    def validate_channel(cls, v):
        if not v.startswith('#'):
            raise ValueError("Le canal doit commencer par #")
        if len(v) < 2:
            raise ValueError("Le canal doit avoir au moins 1 caractère après #")
        return v
    
    @validator('title')
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError("Le titre ne peut pas être vide")
        return v.strip()
    
    @validator('raw_data')
    def validate_raw_data(cls, v):
        # Limiter la taille des données brutes
        serialized = json.dumps(v)
        if len(serialized) > 10000:  # 10KB max
            raise ValueError("Les données brutes ne peuvent pas dépasser 10KB")
        return v

class AlertTemplateAPI(BaseModel):
    """Modèle API pour template d'alerte."""
    name: str = Field(..., min_length=1, max_length=100)
    alert_type: AlertType = Field(...)
    severity: AlertSeverity = Field(...)
    locale: Locale = Field(...)
    channel: NotificationChannel = Field(...)
    title_template: str = Field(..., min_length=1, max_length=500)
    message_template: str = Field(..., min_length=1, max_length=4000)
    variables: List[str] = Field(default_factory=list)
    
    @validator('title_template', 'message_template')
    def validate_templates(cls, v):
        # Validation basique des templates Jinja2
        if '{{' in v and '}}' not in v:
            raise ValueError("Template mal formé: {{ sans }}")
        return v

# ============================================================================================
# FACTORY ET BUILDERS
# ============================================================================================

class AlertFactory:
    """Factory pour créer des alertes optimisées."""
    
    @staticmethod
    def create_slack_alert(
        tenant_id: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        channel: str,
        alert_type: AlertType = AlertType.SYSTEM,
        locale: Locale = Locale.FR,
        **kwargs
    ) -> SlackAlert:
        """Crée une alerte Slack optimisée."""
        
        # Génération d'ID corrélation si non fourni
        correlation_id = kwargs.get('correlation_id') or f"corr_{uuid.uuid4().hex[:16]}"
        
        # Calcul automatique du SLA deadline
        sla_mapping = {
            AlertSeverity.CRITICAL: timedelta(minutes=15),
            AlertSeverity.HIGH: timedelta(hours=1),
            AlertSeverity.MEDIUM: timedelta(hours=4),
            AlertSeverity.LOW: timedelta(hours=24),
            AlertSeverity.INFO: timedelta(days=7),
            AlertSeverity.DEBUG: None
        }
        
        sla_deadline = None
        if severity in sla_mapping and sla_mapping[severity]:
            sla_deadline = datetime.utcnow() + sla_mapping[severity]
        
        alert = SlackAlert(
            tenant_id=tenant_id,
            environment=kwargs.get('environment', Environment.DEV),
            severity=severity,
            alert_type=alert_type,
            title=title,
            message=message,
            channel=channel,
            locale=locale,
            correlation_id=correlation_id,
            sla_deadline=sla_deadline,
            source_system=kwargs.get('source_system', 'achiri-ai-agent'),
            raw_data=kwargs.get('raw_data', {}),
            security_level=kwargs.get('security_level', SecurityLevel.INTERNAL)
        )
        
        # Ajout d'audit automatique
        alert.add_audit_entry(
            action="alert_created",
            user=kwargs.get('created_by', 'system'),
            details={"factory": "AlertFactory.create_slack_alert"}
        )
        
        return alert

    @staticmethod
    def create_from_webhook(webhook_data: Dict[str, Any]) -> SlackAlert:
        """Crée une alerte à partir d'un webhook Alertmanager."""
        
        # Extraction des données du webhook
        alerts = webhook_data.get('alerts', [])
        if not alerts:
            raise ValueError("Aucune alerte trouvée dans le webhook")
        
        first_alert = alerts[0]
        labels = first_alert.get('labels', {})
        annotations = first_alert.get('annotations', {})
        
        # Mapping de la sévérité
        severity_mapping = {
            'critical': AlertSeverity.CRITICAL,
            'warning': AlertSeverity.HIGH,
            'info': AlertSeverity.INFO
        }
        
        severity = severity_mapping.get(
            labels.get('severity', 'info').lower(),
            AlertSeverity.INFO
        )
        
        # Création de l'alerte
        return AlertFactory.create_slack_alert(
            tenant_id=labels.get('tenant_id', 'default'),
            severity=severity,
            title=annotations.get('summary', 'Alerte Prometheus'),
            message=annotations.get('description', 'Alerte sans description'),
            channel=labels.get('slack_channel', '#alerts'),
            alert_type=AlertType.SYSTEM,
            source_system='prometheus',
            raw_data=webhook_data,
            correlation_id=first_alert.get('fingerprint')
        )

# ============================================================================================
# UTILITAIRES ET HELPERS
# ============================================================================================

def serialize_for_cache(obj: BaseEntity) -> str:
    """Sérialise un objet pour le cache."""
    return json.dumps(obj.to_dict(), default=str, sort_keys=True)

def deserialize_from_cache(data: str, model_class: Type[BaseEntity]) -> BaseEntity:
    """Désérialise un objet depuis le cache."""
    obj_data = json.loads(data)
    return model_class(**obj_data)

def validate_alert_data(data: Dict[str, Any]) -> List[str]:
    """Valide les données d'une alerte et retourne les erreurs."""
    errors = []
    
    required_fields = ['tenant_id', 'severity', 'title', 'message', 'channel']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Champ requis manquant: {field}")
    
    if 'severity' in data:
        try:
            AlertSeverity(data['severity'])
        except ValueError:
            errors.append(f"Sévérité invalide: {data['severity']}")
    
    if 'channel' in data and not data['channel'].startswith('#'):
        errors.append("Le canal doit commencer par #")
    
    return errors

def calculate_alert_priority(
    severity: AlertSeverity,
    age_hours: float,
    tenant_priority: int = 100
) -> int:
    """Calcule la priorité d'une alerte."""
    severity_weights = {
        AlertSeverity.CRITICAL: 1000,
        AlertSeverity.HIGH: 750,
        AlertSeverity.MEDIUM: 500,
        AlertSeverity.LOW: 250,
        AlertSeverity.INFO: 100,
        AlertSeverity.DEBUG: 50
    }
    
    base_score = severity_weights.get(severity, 0)
    age_factor = min(age_hours * 5, 200)  # Max 200 pour l'âge
    tenant_factor = min(tenant_priority, 500)  # Max 500 pour le tenant
    
    return int(base_score + age_factor + tenant_factor)

# Export de tous les modèles
__all__ = [
    # Enums et types
    "AlertSeverity", "Environment", "Locale", "NotificationChannel", 
    "AlertType", "SecurityLevel", "ComplianceStandard",
    
    # Modèles de base
    "BaseEntity", "AuditableMixin", "EncryptableMixin",
    
    # Modèles principaux
    "SlackAlert", "AlertTemplate", "TenantFixture", "SecurityPolicy",
    "PerformanceMetric", "TenantMetrics", "MLModel", "MLPrediction",
    "AnomalyEvent", "AuditEvent", "MonitoringEvent", "BusinessRule",
    "SLAPolicy", "EscalationEvent", "HealthCheckResult", "BackupRecord",
    "RecoveryRecord",
    
    # Modèles API
    "SlackAlertAPI", "AlertTemplateAPI",
    
    # Factory et utilitaires
    "AlertFactory", "serialize_for_cache", "deserialize_from_cache",
    "validate_alert_data", "calculate_alert_priority"
]
