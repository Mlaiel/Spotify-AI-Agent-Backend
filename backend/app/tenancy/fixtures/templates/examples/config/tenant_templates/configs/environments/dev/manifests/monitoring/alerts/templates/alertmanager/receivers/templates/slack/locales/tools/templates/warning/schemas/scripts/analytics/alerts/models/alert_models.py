"""
Modèles de données pour les alertes - Spotify AI Agent
Modèles Pydantic pour validation et sérialisation des alertes
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4

class AlertSeverity(str, Enum):
    """Niveaux de gravité des alertes"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertStatus(str, Enum):
    """États des alertes"""
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"
    PENDING = "pending"

class AlertSource(str, Enum):
    """Sources d'alertes"""
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    CUSTOM = "custom"

class AlertCategory(str, Enum):
    """Catégories d'alertes"""
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    SECURITY = "security"
    BUSINESS = "business"
    OPERATIONAL = "operational"

class AlertMetrics(BaseModel):
    """Métriques associées à une alerte"""
    cpu_usage: Optional[float] = Field(None, ge=0, le=100, description="Utilisation CPU (%)")
    memory_usage: Optional[float] = Field(None, ge=0, le=100, description="Utilisation mémoire (%)")
    disk_usage: Optional[float] = Field(None, ge=0, le=100, description="Utilisation disque (%)")
    response_time: Optional[float] = Field(None, ge=0, description="Temps de réponse (ms)")
    error_rate: Optional[float] = Field(None, ge=0, le=100, description="Taux d'erreur (%)")
    request_rate: Optional[float] = Field(None, ge=0, description="Taux de requêtes (req/s)")
    latency_p95: Optional[float] = Field(None, ge=0, description="Latence P95 (ms)")
    latency_p99: Optional[float] = Field(None, ge=0, description="Latence P99 (ms)")
    throughput: Optional[float] = Field(None, ge=0, description="Débit (MB/s)")
    connections: Optional[int] = Field(None, ge=0, description="Nombre de connexions")
    custom_metrics: Dict[str, float] = Field(default_factory=dict, description="Métriques personnalisées")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AlertAnnotation(BaseModel):
    """Annotations d'alerte"""
    summary: Optional[str] = Field(None, description="Résumé de l'alerte")
    description: Optional[str] = Field(None, description="Description détaillée")
    runbook_url: Optional[str] = Field(None, description="URL du runbook")
    dashboard_url: Optional[str] = Field(None, description="URL du dashboard")
    documentation_url: Optional[str] = Field(None, description="URL de documentation")
    escalation_policy: Optional[str] = Field(None, description="Politique d'escalade")
    remediation_steps: List[str] = Field(default_factory=list, description="Étapes de résolution")
    impact_assessment: Optional[str] = Field(None, description="Évaluation d'impact")
    business_context: Optional[str] = Field(None, description="Contexte métier")

class AlertEvent(BaseModel):
    """Modèle principal d'événement d'alerte"""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Identifiant unique")
    timestamp: datetime = Field(default_factory=datetime.now, description="Horodatage")
    severity: AlertSeverity = Field(..., description="Niveau de gravité")
    status: AlertStatus = Field(default=AlertStatus.FIRING, description="État de l'alerte")
    source: AlertSource = Field(..., description="Source de l'alerte")
    category: AlertCategory = Field(..., description="Catégorie de l'alerte")
    
    # Identification
    service: str = Field(..., min_length=1, description="Service concerné")
    component: Optional[str] = Field(None, description="Composant spécifique")
    environment: str = Field(default="production", description="Environnement")
    region: Optional[str] = Field(None, description="Région géographique")
    datacenter: Optional[str] = Field(None, description="Centre de données")
    
    # Contenu
    title: str = Field(..., min_length=1, description="Titre de l'alerte")
    message: str = Field(..., min_length=1, description="Message d'alerte")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels Prometheus")
    annotations: AlertAnnotation = Field(default_factory=AlertAnnotation, description="Annotations")
    
    # Métriques et données
    metrics: AlertMetrics = Field(default_factory=AlertMetrics, description="Métriques")
    threshold_value: Optional[float] = Field(None, description="Valeur de seuil")
    current_value: Optional[float] = Field(None, description="Valeur actuelle")
    
    # Métadonnées
    fingerprint: Optional[str] = Field(None, description="Empreinte unique")
    correlation_id: Optional[str] = Field(None, description="ID de corrélation")
    incident_id: Optional[str] = Field(None, description="ID d'incident")
    alert_rule: Optional[str] = Field(None, description="Règle d'alerte")
    
    # Timing
    starts_at: Optional[datetime] = Field(None, description="Début de l'alerte")
    ends_at: Optional[datetime] = Field(None, description="Fin de l'alerte")
    resolved_at: Optional[datetime] = Field(None, description="Moment de résolution")
    acknowledged_at: Optional[datetime] = Field(None, description="Moment d'acquittement")
    acknowledged_by: Optional[str] = Field(None, description="Acquitté par")
    
    # Enrichissement
    tags: List[str] = Field(default_factory=list, description="Tags personnalisés")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contexte additionnel")
    external_links: List[str] = Field(default_factory=list, description="Liens externes")
    
    @validator('severity')
    def validate_severity(cls, v):
        """Validation du niveau de gravité"""
        if v not in AlertSeverity:
            raise ValueError(f"Gravité invalide: {v}")
        return v
    
    @validator('service')
    def validate_service(cls, v):
        """Validation du nom de service"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Le service ne peut pas être vide")
        return v.strip().lower()
    
    @validator('timestamp', 'starts_at', 'ends_at', 'resolved_at', 'acknowledged_at')
    def validate_timestamps(cls, v):
        """Validation des horodatages"""
        if v and v > datetime.now():
            # Autoriser un léger décalage dans le futur (5 minutes)
            if (v - datetime.now()).total_seconds() > 300:
                raise ValueError("L'horodatage ne peut pas être trop dans le futur")
        return v
    
    @validator('current_value', 'threshold_value')
    def validate_numeric_values(cls, v):
        """Validation des valeurs numériques"""
        if v is not None and (v < 0 or v > 1e12):  # Limite raisonnable
            raise ValueError("Valeur numérique hors limites")
        return v
    
    def to_prometheus_format(self) -> Dict[str, Any]:
        """Conversion au format Prometheus"""
        return {
            "alerts": [{
                "status": self.status.value,
                "labels": {
                    "alertname": self.title,
                    "service": self.service,
                    "severity": self.severity.value,
                    "environment": self.environment,
                    **self.labels
                },
                "annotations": {
                    "summary": self.annotations.summary or self.title,
                    "description": self.annotations.description or self.message,
                    **({k: v for k, v in self.annotations.dict().items() if v is not None})
                },
                "startsAt": self.starts_at.isoformat() if self.starts_at else self.timestamp.isoformat(),
                "endsAt": self.ends_at.isoformat() if self.ends_at else None,
                "fingerprint": self.fingerprint
            }]
        }
    
    def get_display_name(self) -> str:
        """Nom d'affichage pour l'interface"""
        component_part = f".{self.component}" if self.component else ""
        return f"{self.service}{component_part}: {self.title}"
    
    def is_critical(self) -> bool:
        """Vérifie si l'alerte est critique"""
        return self.severity == AlertSeverity.CRITICAL
    
    def is_active(self) -> bool:
        """Vérifie si l'alerte est active"""
        return self.status == AlertStatus.FIRING
    
    def duration(self) -> Optional[float]:
        """Durée de l'alerte en secondes"""
        if self.starts_at:
            end_time = self.ends_at or datetime.now()
            return (end_time - self.starts_at).total_seconds()
        return None
    
    def add_context(self, key: str, value: Any):
        """Ajout de contexte"""
        self.context[key] = value
    
    def add_tag(self, tag: str):
        """Ajout d'un tag"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_metric(self, metric_name: str, value: float):
        """Définition d'une métrique personnalisée"""
        self.metrics.custom_metrics[metric_name] = value
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "id": "alert-12345",
                "timestamp": "2025-01-19T10:00:00Z",
                "severity": "high",
                "status": "firing",
                "source": "prometheus",
                "category": "performance",
                "service": "user-api",
                "component": "authentication",
                "environment": "production",
                "title": "High CPU Usage",
                "message": "CPU usage has exceeded 90% for 5 minutes",
                "labels": {
                    "instance": "user-api-01",
                    "job": "user-api"
                },
                "metrics": {
                    "cpu_usage": 95.5,
                    "response_time": 1200
                },
                "threshold_value": 90.0,
                "current_value": 95.5
            }
        }

class AlertBatch(BaseModel):
    """Lot d'alertes pour traitement en masse"""
    alerts: List[AlertEvent] = Field(..., description="Liste d'alertes")
    batch_id: str = Field(default_factory=lambda: str(uuid4()), description="ID du lot")
    created_at: datetime = Field(default_factory=datetime.now, description="Horodatage de création")
    source_system: str = Field(..., description="Système source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées du lot")
    
    @validator('alerts')
    def validate_alerts_not_empty(cls, v):
        """Validation que le lot n'est pas vide"""
        if not v:
            raise ValueError("Le lot d'alertes ne peut pas être vide")
        return v
    
    def count_by_severity(self) -> Dict[AlertSeverity, int]:
        """Comptage par gravité"""
        counts = {severity: 0 for severity in AlertSeverity}
        for alert in self.alerts:
            counts[alert.severity] += 1
        return counts
    
    def get_critical_alerts(self) -> List[AlertEvent]:
        """Récupération des alertes critiques"""
        return [alert for alert in self.alerts if alert.is_critical()]
    
    def get_services(self) -> List[str]:
        """Liste des services concernés"""
        return list(set(alert.service for alert in self.alerts))

class AlertQuery(BaseModel):
    """Requête de recherche d'alertes"""
    services: Optional[List[str]] = Field(None, description="Services à filtrer")
    severities: Optional[List[AlertSeverity]] = Field(None, description="Gravités à filtrer")
    statuses: Optional[List[AlertStatus]] = Field(None, description="États à filtrer")
    sources: Optional[List[AlertSource]] = Field(None, description="Sources à filtrer")
    categories: Optional[List[AlertCategory]] = Field(None, description="Catégories à filtrer")
    
    start_time: Optional[datetime] = Field(None, description="Début de période")
    end_time: Optional[datetime] = Field(None, description="Fin de période")
    
    search_text: Optional[str] = Field(None, description="Texte de recherche")
    labels: Optional[Dict[str, str]] = Field(None, description="Filtrage par labels")
    
    limit: int = Field(default=100, ge=1, le=10000, description="Limite de résultats")
    offset: int = Field(default=0, ge=0, description="Décalage pour pagination")
    
    sort_by: str = Field(default="timestamp", description="Champ de tri")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$", description="Ordre de tri")
    
    include_resolved: bool = Field(default=True, description="Inclure les alertes résolues")
    include_suppressed: bool = Field(default=False, description="Inclure les alertes supprimées")

class AlertStatistics(BaseModel):
    """Statistiques d'alertes"""
    total_alerts: int = Field(..., description="Nombre total d'alertes")
    alerts_by_severity: Dict[AlertSeverity, int] = Field(..., description="Répartition par gravité")
    alerts_by_status: Dict[AlertStatus, int] = Field(..., description="Répartition par état")
    alerts_by_service: Dict[str, int] = Field(..., description="Répartition par service")
    
    avg_resolution_time: Optional[float] = Field(None, description="Temps moyen de résolution (s)")
    critical_alerts_count: int = Field(..., description="Nombre d'alertes critiques")
    active_alerts_count: int = Field(..., description="Nombre d'alertes actives")
    
    time_period: str = Field(..., description="Période analysée")
    generated_at: datetime = Field(default_factory=datetime.now, description="Horodatage de génération")
    
    def get_critical_percentage(self) -> float:
        """Pourcentage d'alertes critiques"""
        if self.total_alerts == 0:
            return 0.0
        return (self.critical_alerts_count / self.total_alerts) * 100
    
    def get_resolution_rate(self) -> float:
        """Taux de résolution"""
        resolved_count = self.alerts_by_status.get(AlertStatus.RESOLVED, 0)
        if self.total_alerts == 0:
            return 0.0
        return (resolved_count / self.total_alerts) * 100
