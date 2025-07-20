"""
Alert Schema Module
==================

Ce module définit les schémas pour le système d'alerting multi-tenant avancé
avec support des notifications intelligentes, escalation automatique et
intégration avec les systèmes de monitoring externes.
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.networks import HttpUrl


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """États possibles d'une alerte."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    SUPPRESSED = "suppressed"


class AlertCategory(str, Enum):
    """Catégories d'alertes."""
    PERFORMANCE = "performance"
    SECURITY = "security"
    AVAILABILITY = "availability"
    CAPACITY = "capacity"
    COMPLIANCE = "compliance"
    BUSINESS = "business"
    TECHNICAL = "technical"
    BILLING = "billing"


class AlertChannel(str, Enum):
    """Canaux de notification."""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    DISCORD = "discord"
    PHONE = "phone"


class EscalationLevel(str, Enum):
    """Niveaux d'escalation."""
    L1 = "l1"  # First line support
    L2 = "l2"  # Technical specialists
    L3 = "l3"  # Senior engineers
    L4 = "l4"  # Architecture team
    EXECUTIVE = "executive"  # C-level


class AlertCondition(BaseModel):
    """Condition de déclenchement d'alerte."""
    metric_name: str = Field(..., description="Nom de la métrique")
    operator: str = Field(..., regex="^(gt|gte|lt|lte|eq|ne|contains|not_contains)$")
    threshold: Union[float, int, str] = Field(..., description="Seuil de déclenchement")
    duration_minutes: int = Field(5, ge=1, le=1440, description="Durée avant déclenchement")
    evaluation_window: int = Field(1, ge=1, le=60, description="Fenêtre d'évaluation (min)")
    
    class Config:
        schema_extra = {
            "example": {
                "metric_name": "cpu_usage_percent",
                "operator": "gt",
                "threshold": 80.0,
                "duration_minutes": 5,
                "evaluation_window": 1
            }
        }


class AlertAction(BaseModel):
    """Action à exécuter lors d'une alerte."""
    action_type: str = Field(..., regex="^(notification|webhook|script|escalation|auto_scale)$")
    target: str = Field(..., description="Cible de l'action")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = Field(3, ge=0, le=10)
    retry_delay_seconds: int = Field(60, ge=5, le=3600)
    
    class Config:
        schema_extra = {
            "example": {
                "action_type": "notification",
                "target": "slack://alerts-channel",
                "parameters": {
                    "message_template": "🚨 Alert: {title} - {description}",
                    "mention_users": ["@devops", "@oncall"]
                }
            }
        }


class AlertEscalation(BaseModel):
    """Configuration d'escalation d'alerte."""
    level: EscalationLevel = Field(..., description="Niveau d'escalation")
    delay_minutes: int = Field(30, ge=1, le=1440, description="Délai avant escalation")
    channels: List[AlertChannel] = Field(..., description="Canaux d'escalation")
    recipients: List[str] = Field(..., description="Destinataires")
    conditions: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "level": "l2",
                "delay_minutes": 30,
                "channels": ["email", "slack"],
                "recipients": ["l2-team@company.com", "#escalations"],
                "conditions": {"severity": ["critical", "high"]}
            }
        }


class AlertSuppression(BaseModel):
    """Configuration de suppression d'alerte."""
    enabled: bool = Field(False, description="Suppression activée")
    start_time: Optional[str] = Field(None, regex="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    end_time: Optional[str] = Field(None, regex="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    days_of_week: List[int] = Field(default_factory=list, description="Jours de la semaine (0-6)")
    reason: Optional[str] = Field(None, max_length=200)
    
    @validator('days_of_week')
    def validate_days(cls, v):
        """Valide les jours de la semaine."""
        for day in v:
            if not 0 <= day <= 6:
                raise ValueError("Day must be between 0 (Monday) and 6 (Sunday)")
        return v


class AlertMetrics(BaseModel):
    """Métriques associées à une alerte."""
    response_time_ms: Optional[float] = Field(None, ge=0)
    acknowledgment_time_ms: Optional[float] = Field(None, ge=0)
    resolution_time_ms: Optional[float] = Field(None, ge=0)
    escalation_count: int = Field(0, ge=0)
    false_positive: bool = Field(False)
    user_satisfaction: Optional[int] = Field(None, ge=1, le=5)


class AlertSchema(BaseModel):
    """
    Schéma principal pour les alertes système avec support complet
    du multi-tenant et des fonctionnalités avancées d'escalation.
    """
    # Identifiants et métadonnées
    alert_id: str = Field(default_factory=lambda: str(uuid4()), description="ID unique de l'alerte")
    tenant_id: str = Field(..., description="ID du tenant")
    name: str = Field(..., min_length=3, max_length=100, description="Nom de l'alerte")
    title: str = Field(..., min_length=5, max_length=200, description="Titre de l'alerte")
    description: str = Field(..., min_length=10, max_length=1000, description="Description détaillée")
    
    # Classification
    severity: AlertSeverity = Field(..., description="Niveau de sévérité")
    category: AlertCategory = Field(..., description="Catégorie de l'alerte")
    status: AlertStatus = Field(AlertStatus.OPEN, description="État actuel")
    
    # Conditions et configuration
    conditions: List[AlertCondition] = Field(..., min_items=1, description="Conditions de déclenchement")
    actions: List[AlertAction] = Field(default_factory=list, description="Actions à exécuter")
    escalations: List[AlertEscalation] = Field(default_factory=list, description="Escalations configurées")
    suppression: AlertSuppression = Field(default_factory=AlertSuppression, description="Config suppression")
    
    # Canaux et notifications
    notification_channels: List[AlertChannel] = Field(default_factory=list)
    recipients: List[str] = Field(default_factory=list, description="Destinataires principaux")
    webhook_urls: List[HttpUrl] = Field(default_factory=list, description="URLs webhook")
    
    # Timing et planification
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    triggered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Configuration avancée
    enabled: bool = Field(True, description="Alerte activée")
    auto_resolve: bool = Field(False, description="Résolution automatique")
    auto_resolve_minutes: int = Field(60, ge=5, le=1440, description="Délai auto-résolution")
    cooldown_minutes: int = Field(5, ge=1, le=60, description="Période de refroidissement")
    max_notifications: int = Field(10, ge=1, le=100, description="Nombre max de notifications")
    
    # Métadonnées et contexte
    source: str = Field("system", description="Source de l'alerte")
    environment: str = Field("production", description="Environnement cible")
    service: Optional[str] = Field(None, description="Service concerné")
    component: Optional[str] = Field(None, description="Composant concerné")
    runbook_url: Optional[HttpUrl] = Field(None, description="URL du runbook")
    
    # Tags et métadonnées
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags personnalisés")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels pour routage")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Annotations")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contexte additionnel")
    
    # Métriques et analytics
    metrics: AlertMetrics = Field(default_factory=AlertMetrics, description="Métriques de l'alerte")
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "name": "high_cpu_usage",
                "title": "High CPU Usage Detected",
                "description": "CPU usage has exceeded 80% for more than 5 minutes",
                "severity": "high",
                "category": "performance",
                "conditions": [
                    {
                        "metric_name": "cpu_usage_percent",
                        "operator": "gt",
                        "threshold": 80.0,
                        "duration_minutes": 5
                    }
                ],
                "notification_channels": ["email", "slack"],
                "recipients": ["ops-team@company.com", "#alerts"]
            }
        }
    
    @validator('updated_at', always=True)
    def set_updated_at(cls, v):
        """Met à jour automatiquement le timestamp."""
        return v or datetime.now(timezone.utc)
    
    @validator('auto_resolve_minutes')
    def validate_auto_resolve(cls, v, values):
        """Valide la configuration d'auto-résolution."""
        if values.get('auto_resolve') and v < 5:
            raise ValueError("Auto-resolve delay must be at least 5 minutes")
        return v
    
    @root_validator
    def validate_escalations(cls, values):
        """Valide la configuration des escalations."""
        escalations = values.get('escalations', [])
        if not escalations:
            return values
        
        # Vérifier l'ordre des niveaux d'escalation
        levels = [esc.level for esc in escalations]
        expected_order = [EscalationLevel.L1, EscalationLevel.L2, EscalationLevel.L3, EscalationLevel.L4]
        
        for i, level in enumerate(levels[:-1]):
            if level in expected_order and levels[i+1] in expected_order:
                if expected_order.index(level) >= expected_order.index(levels[i+1]):
                    raise ValueError("Escalation levels must be in ascending order")
        
        return values
    
    @root_validator
    def validate_severity_escalations(cls, values):
        """Valide que les alertes critiques ont des escalations."""
        severity = values.get('severity')
        escalations = values.get('escalations', [])
        
        if severity == AlertSeverity.CRITICAL and not escalations:
            raise ValueError("Critical alerts must have escalation configured")
        
        return values
    
    def get_effective_recipients(self) -> List[str]:
        """Retourne la liste complète des destinataires incluant les escalations."""
        all_recipients = self.recipients.copy()
        for escalation in self.escalations:
            all_recipients.extend(escalation.recipients)
        return list(set(all_recipients))
    
    def is_suppressed(self, current_time: Optional[datetime] = None) -> bool:
        """Vérifie si l'alerte est actuellement supprimée."""
        if not self.suppression.enabled:
            return False
        
        current_time = current_time or datetime.now(timezone.utc)
        
        # Vérifier les jours de la semaine
        if self.suppression.days_of_week:
            weekday = current_time.weekday()
            if weekday not in self.suppression.days_of_week:
                return False
        
        # Vérifier les heures
        if self.suppression.start_time and self.suppression.end_time:
            current_time_str = current_time.strftime("%H:%M")
            if self.suppression.start_time <= current_time_str <= self.suppression.end_time:
                return True
        
        return False
    
    def calculate_priority_score(self) -> float:
        """Calcule un score de priorité pour le tri des alertes."""
        severity_weights = {
            AlertSeverity.CRITICAL: 100,
            AlertSeverity.HIGH: 80,
            AlertSeverity.MEDIUM: 60,
            AlertSeverity.LOW: 40,
            AlertSeverity.INFO: 20
        }
        
        base_score = severity_weights.get(self.severity, 0)
        
        # Bonus pour les escalations configurées
        if self.escalations:
            base_score += 10
        
        # Malus pour les alertes anciennes
        if self.triggered_at:
            age_hours = (datetime.now(timezone.utc) - self.triggered_at).total_seconds() / 3600
            base_score += min(age_hours * 2, 20)  # Max 20 points pour l'âge
        
        return base_score


class TenantAlertSchema(AlertSchema):
    """
    Extension du schéma d'alerte avec des fonctionnalités spécifiques
    au multi-tenant et à la gestion avancée des permissions.
    """
    tenant_specific_config: Dict[str, Any] = Field(default_factory=dict)
    tenant_permissions: Dict[str, List[str]] = Field(default_factory=dict)
    cross_tenant_visibility: bool = Field(False, description="Visibilité cross-tenant")
    delegation_rules: List[Dict[str, Any]] = Field(default_factory=list)
    sla_impact: bool = Field(False, description="Impact sur les SLA")
    business_impact: Optional[str] = Field(None, max_length=500)
    customer_facing: bool = Field(False, description="Alerte visible client")
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "name": "tenant_quota_exceeded",
                "title": "Storage Quota Exceeded",
                "description": "Tenant has exceeded their allocated storage quota",
                "severity": "high",
                "category": "capacity",
                "tenant_specific_config": {
                    "quota_type": "storage",
                    "current_usage": "950GB",
                    "limit": "1TB"
                },
                "sla_impact": True,
                "customer_facing": True
            }
        }


class AlertTemplateSchema(BaseModel):
    """Template pour la création d'alertes standardisées."""
    template_id: str = Field(..., description="ID du template")
    name: str = Field(..., description="Nom du template")
    description: str = Field(..., description="Description du template")
    category: AlertCategory = Field(..., description="Catégorie")
    default_severity: AlertSeverity = Field(..., description="Sévérité par défaut")
    condition_templates: List[Dict[str, Any]] = Field(..., description="Templates de conditions")
    action_templates: List[Dict[str, Any]] = Field(default_factory=list)
    escalation_templates: List[Dict[str, Any]] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variables du template")
    tenant_types: List[str] = Field(default_factory=list, description="Types de tenant supportés")
    
    class Config:
        schema_extra = {
            "example": {
                "template_id": "high_cpu_template",
                "name": "High CPU Usage Template",
                "description": "Standard template for CPU usage alerts",
                "category": "performance",
                "default_severity": "high",
                "condition_templates": [
                    {
                        "metric_name": "cpu_usage_percent",
                        "operator": "gt",
                        "threshold": "{{cpu_threshold}}",
                        "duration_minutes": "{{duration}}"
                    }
                ],
                "variables": {
                    "cpu_threshold": 80.0,
                    "duration": 5
                }
            }
        }


class AlertSummarySchema(BaseModel):
    """Résumé des alertes pour les dashboards et rapports."""
    tenant_id: str = Field(..., description="ID du tenant")
    period_start: datetime = Field(..., description="Début de la période")
    period_end: datetime = Field(..., description="Fin de la période")
    total_alerts: int = Field(0, ge=0)
    alerts_by_severity: Dict[AlertSeverity, int] = Field(default_factory=dict)
    alerts_by_category: Dict[AlertCategory, int] = Field(default_factory=dict)
    alerts_by_status: Dict[AlertStatus, int] = Field(default_factory=dict)
    average_response_time_minutes: Optional[float] = Field(None, ge=0)
    average_resolution_time_minutes: Optional[float] = Field(None, ge=0)
    escalation_rate: float = Field(0, ge=0, le=100)
    false_positive_rate: float = Field(0, ge=0, le=100)
    sla_breaches: int = Field(0, ge=0)
    top_alert_sources: List[Dict[str, Union[str, int]]] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "period_start": "2024-01-01T00:00:00Z",
                "period_end": "2024-01-31T23:59:59Z",
                "total_alerts": 245,
                "alerts_by_severity": {
                    "critical": 12,
                    "high": 45,
                    "medium": 123,
                    "low": 65
                },
                "average_response_time_minutes": 8.5,
                "escalation_rate": 15.2
            }
        }
