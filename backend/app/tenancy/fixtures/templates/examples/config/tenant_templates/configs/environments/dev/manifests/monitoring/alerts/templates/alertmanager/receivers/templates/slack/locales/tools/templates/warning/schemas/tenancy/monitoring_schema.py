"""
Monitoring Configuration Schema Module
=====================================

Ce module définit les schémas pour la configuration du monitoring multi-tenant
avec support des métriques personnalisées, dashboards dynamiques et intégration
avec les systèmes de observabilité modernes (Prometheus, Grafana, OpenTelemetry).
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.networks import HttpUrl, IPvAnyAddress


class MetricType(str, Enum):
    """Types de métriques supportées."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class MetricSource(str, Enum):
    """Sources de métriques."""
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    BUSINESS = "business"
    SYNTHETIC = "synthetic"
    USER_DEFINED = "user_defined"
    EXTERNAL = "external"


class AggregationMethod(str, Enum):
    """Méthodes d'agrégation."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    RATE = "rate"
    PERCENTILE = "percentile"
    STDDEV = "stddev"


class AlertConditionOperator(str, Enum):
    """Opérateurs pour conditions d'alerte."""
    GT = "gt"          # greater than
    GTE = "gte"        # greater than or equal
    LT = "lt"          # less than
    LTE = "lte"        # less than or equal
    EQ = "eq"          # equal
    NEQ = "neq"        # not equal
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    NOT_REGEX = "not_regex"


class DashboardType(str, Enum):
    """Types de dashboards."""
    OVERVIEW = "overview"
    DETAILED = "detailed"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    CUSTOM = "custom"
    REAL_TIME = "real_time"


class RetentionPeriod(str, Enum):
    """Périodes de rétention des données."""
    HOUR_1 = "1h"
    HOURS_6 = "6h"
    HOURS_24 = "24h"
    DAYS_7 = "7d"
    DAYS_30 = "30d"
    DAYS_90 = "90d"
    DAYS_365 = "365d"
    DAYS_1095 = "1095d"  # 3 ans


class MonitoringMetric(BaseModel):
    """Définition d'une métrique de monitoring."""
    metric_id: str = Field(..., description="ID unique de la métrique")
    name: str = Field(..., description="Nom de la métrique")
    display_name: str = Field(..., description="Nom d'affichage")
    description: str = Field(..., description="Description")
    
    # Type et source
    metric_type: MetricType = Field(..., description="Type de métrique")
    source: MetricSource = Field(..., description="Source de la métrique")
    query: str = Field(..., description="Requête pour collecter la métrique")
    
    # Configuration
    unit: str = Field(..., description="Unité de mesure")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags par défaut")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels")
    
    # Agrégation et sampling
    aggregation_method: AggregationMethod = Field(AggregationMethod.AVG)
    sampling_interval_seconds: int = Field(60, ge=1, le=3600)
    retention_period: RetentionPeriod = Field(RetentionPeriod.DAYS_30)
    
    # Seuils et alertes
    warning_threshold: Optional[float] = Field(None, description="Seuil d'avertissement")
    critical_threshold: Optional[float] = Field(None, description="Seuil critique")
    baseline_value: Optional[float] = Field(None, description="Valeur de référence")
    
    # Configuration avancée
    enabled: bool = Field(True, description="Métrique activée")
    tenant_specific: bool = Field(True, description="Spécifique au tenant")
    business_impact: bool = Field(False, description="Impact business")
    
    class Config:
        schema_extra = {
            "example": {
                "metric_id": "cpu_usage_percent",
                "name": "cpu_usage_percent",
                "display_name": "CPU Usage (%)",
                "description": "Percentage of CPU utilization",
                "metric_type": "gauge",
                "source": "infrastructure",
                "query": "avg(cpu_usage_percent) by (instance)",
                "unit": "percent",
                "warning_threshold": 75.0,
                "critical_threshold": 90.0
            }
        }


class DashboardPanel(BaseModel):
    """Panel d'un dashboard."""
    panel_id: str = Field(..., description="ID du panel")
    title: str = Field(..., description="Titre du panel")
    description: Optional[str] = Field(None, description="Description")
    
    # Configuration du panel
    panel_type: str = Field(..., regex="^(graph|table|stat|gauge|heatmap|logs|alert_list)$")
    metrics: List[str] = Field(..., description="IDs des métriques affichées")
    time_range: str = Field("1h", description="Plage de temps par défaut")
    
    # Position et taille
    position: Dict[str, int] = Field(..., description="Position (x, y, width, height)")
    
    # Visualisation
    visualization_config: Dict[str, Any] = Field(default_factory=dict)
    color_scheme: Optional[str] = Field(None, description="Schéma de couleurs")
    
    # Interactivité
    drill_down_enabled: bool = Field(False, description="Drill-down activé")
    export_enabled: bool = Field(True, description="Export activé")
    
    class Config:
        schema_extra = {
            "example": {
                "panel_id": "cpu_usage_panel",
                "title": "CPU Usage Over Time",
                "panel_type": "graph",
                "metrics": ["cpu_usage_percent"],
                "position": {"x": 0, "y": 0, "width": 12, "height": 8},
                "visualization_config": {
                    "legend": True,
                    "grid": True,
                    "threshold_lines": [75, 90]
                }
            }
        }


class MonitoringDashboard(BaseModel):
    """Configuration d'un dashboard de monitoring."""
    dashboard_id: str = Field(..., description="ID du dashboard")
    name: str = Field(..., description="Nom du dashboard")
    description: Optional[str] = Field(None, description="Description")
    
    # Type et classification
    dashboard_type: DashboardType = Field(..., description="Type de dashboard")
    category: str = Field(..., description="Catégorie")
    tags: List[str] = Field(default_factory=list, description="Tags")
    
    # Contenu
    panels: List[DashboardPanel] = Field(..., description="Panels du dashboard")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variables du dashboard")
    
    # Configuration
    refresh_interval_seconds: int = Field(30, ge=5, le=3600)
    auto_refresh: bool = Field(True, description="Rafraîchissement automatique")
    shared: bool = Field(False, description="Dashboard partagé")
    
    # Permissions
    viewers: List[str] = Field(default_factory=list, description="Utilisateurs/rôles autorisés")
    editors: List[str] = Field(default_factory=list, description="Editeurs autorisés")
    
    # Métadonnées
    created_by: str = Field(..., description="Créé par")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    version: str = Field("1.0", description="Version du dashboard")
    
    class Config:
        schema_extra = {
            "example": {
                "dashboard_id": "system_overview",
                "name": "System Overview",
                "dashboard_type": "overview",
                "category": "infrastructure",
                "panels": [
                    {
                        "panel_id": "cpu_panel",
                        "title": "CPU Usage",
                        "panel_type": "graph",
                        "metrics": ["cpu_usage_percent"],
                        "position": {"x": 0, "y": 0, "width": 6, "height": 8}
                    }
                ],
                "refresh_interval_seconds": 30,
                "created_by": "admin@company.com"
            }
        }


class MonitoringTarget(BaseModel):
    """Cible de monitoring (service, instance, etc.)."""
    target_id: str = Field(..., description="ID de la cible")
    name: str = Field(..., description="Nom de la cible")
    target_type: str = Field(..., regex="^(service|instance|database|api|custom)$")
    
    # Connexion
    endpoint: HttpUrl = Field(..., description="Endpoint de monitoring")
    health_check_path: str = Field("/health", description="Path health check")
    metrics_path: str = Field("/metrics", description="Path des métriques")
    
    # Configuration
    scrape_interval_seconds: int = Field(30, ge=5, le=300)
    timeout_seconds: int = Field(10, ge=1, le=60)
    
    # Métadonnées
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels")
    environment: str = Field("production", description="Environnement")
    region: Optional[str] = Field(None, description="Région")
    
    # État
    enabled: bool = Field(True, description="Monitoring activé")
    last_seen: Optional[datetime] = None
    status: str = Field("unknown", regex="^(up|down|unknown)$")
    
    class Config:
        schema_extra = {
            "example": {
                "target_id": "api_service_prod",
                "name": "API Service Production",
                "target_type": "service",
                "endpoint": "https://api.company.com",
                "labels": {
                    "service": "api",
                    "team": "backend",
                    "version": "v2.1.0"
                },
                "environment": "production"
            }
        }


class AlertRule(BaseModel):
    """Règle d'alerte pour le monitoring."""
    rule_id: str = Field(..., description="ID de la règle")
    name: str = Field(..., description="Nom de la règle")
    description: str = Field(..., description="Description")
    
    # Condition
    metric_name: str = Field(..., description="Nom de la métrique")
    operator: AlertConditionOperator = Field(..., description="Opérateur")
    threshold: Union[float, int, str] = Field(..., description="Seuil")
    
    # Configuration temporelle
    duration_minutes: int = Field(5, ge=1, le=1440, description="Durée avant déclenchement")
    evaluation_interval_minutes: int = Field(1, ge=1, le=60, description="Intervalle d'évaluation")
    
    # Labels et annotations
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Annotations")
    
    # Configuration
    severity: str = Field(..., regex="^(critical|warning|info)$")
    enabled: bool = Field(True, description="Règle activée")
    
    class Config:
        schema_extra = {
            "example": {
                "rule_id": "high_cpu_alert",
                "name": "High CPU Usage",
                "description": "Alert when CPU usage exceeds 80%",
                "metric_name": "cpu_usage_percent",
                "operator": "gt",
                "threshold": 80.0,
                "duration_minutes": 5,
                "severity": "warning"
            }
        }


class MonitoringConfigSchema(BaseModel):
    """
    Schéma principal de configuration du monitoring multi-tenant
    avec support complet des métriques, dashboards et alertes.
    """
    # Identifiants
    config_id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str = Field(..., description="ID du tenant")
    name: str = Field(..., description="Nom de la configuration")
    description: Optional[str] = Field(None, description="Description")
    
    # Configuration générale
    enabled: bool = Field(True, description="Monitoring activé")
    collection_interval_seconds: int = Field(60, ge=5, le=3600)
    retention_days: int = Field(30, ge=1, le=365)
    
    # Métriques
    metrics: List[MonitoringMetric] = Field(default_factory=list, description="Métriques configurées")
    custom_metrics: List[MonitoringMetric] = Field(default_factory=list, description="Métriques personnalisées")
    
    # Dashboards
    dashboards: List[MonitoringDashboard] = Field(default_factory=list, description="Dashboards")
    default_dashboard: Optional[str] = Field(None, description="Dashboard par défaut")
    
    # Cibles de monitoring
    targets: List[MonitoringTarget] = Field(default_factory=list, description="Cibles de monitoring")
    
    # Règles d'alerte
    alert_rules: List[AlertRule] = Field(default_factory=list, description="Règles d'alerte")
    
    # Configuration des exports
    prometheus_config: Dict[str, Any] = Field(default_factory=dict, description="Config Prometheus")
    grafana_config: Dict[str, Any] = Field(default_factory=dict, description="Config Grafana")
    opentelemetry_config: Dict[str, Any] = Field(default_factory=dict, description="Config OpenTelemetry")
    
    # Intégrations externes
    external_integrations: Dict[str, Any] = Field(default_factory=dict, description="Intégrations externes")
    webhook_endpoints: List[HttpUrl] = Field(default_factory=list, description="Endpoints webhook")
    
    # Sécurité et accès
    access_control: Dict[str, List[str]] = Field(default_factory=dict, description="Contrôle d'accès")
    api_keys: Dict[str, str] = Field(default_factory=dict, description="Clés API")
    
    # Performance et optimisation
    sampling_rate: float = Field(1.0, ge=0.0, le=1.0, description="Taux d'échantillonnage")
    batch_size: int = Field(100, ge=1, le=1000, description="Taille des lots")
    compression_enabled: bool = Field(True, description="Compression activée")
    
    # Métadonnées
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    version: str = Field("1.0", description="Version de la configuration")
    
    # Tags et classification
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags")
    environment: str = Field("production", description="Environnement")
    
    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
        schema_extra = {
            "example": {
                "tenant_id": "enterprise_001",
                "name": "Production Monitoring Config",
                "description": "Complete monitoring configuration for production environment",
                "collection_interval_seconds": 30,
                "retention_days": 90,
                "metrics": [
                    {
                        "metric_id": "cpu_usage",
                        "name": "cpu_usage_percent",
                        "display_name": "CPU Usage (%)",
                        "description": "CPU utilization percentage",
                        "metric_type": "gauge",
                        "source": "infrastructure",
                        "query": "avg(cpu_usage_percent)",
                        "unit": "percent"
                    }
                ],
                "targets": [
                    {
                        "target_id": "api_service",
                        "name": "API Service",
                        "target_type": "service",
                        "endpoint": "https://api.company.com"
                    }
                ]
            }
        }
    
    @validator('updated_at', always=True)
    def set_updated_at(cls, v):
        """Met à jour automatiquement le timestamp."""
        return v or datetime.now(timezone.utc)
    
    @validator('collection_interval_seconds')
    def validate_collection_interval(cls, v):
        """Valide l'intervalle de collecte."""
        if v < 5:
            raise ValueError("Collection interval must be at least 5 seconds")
        return v
    
    @validator('sampling_rate')
    def validate_sampling_rate(cls, v):
        """Valide le taux d'échantillonnage."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Sampling rate must be between 0.0 and 1.0")
        return v
    
    @root_validator
    def validate_dashboard_metrics(cls, values):
        """Valide que les métriques des dashboards existent."""
        metrics = values.get('metrics', [])
        custom_metrics = values.get('custom_metrics', [])
        dashboards = values.get('dashboards', [])
        
        available_metrics = {m.metric_id for m in metrics + custom_metrics}
        
        for dashboard in dashboards:
            for panel in dashboard.panels:
                for metric_id in panel.metrics:
                    if metric_id not in available_metrics:
                        raise ValueError(f"Dashboard panel references unknown metric: {metric_id}")
        
        return values
    
    @root_validator
    def validate_alert_rules_metrics(cls, values):
        """Valide que les règles d'alerte référencent des métriques existantes."""
        metrics = values.get('metrics', [])
        custom_metrics = values.get('custom_metrics', [])
        alert_rules = values.get('alert_rules', [])
        
        available_metrics = {m.name for m in metrics + custom_metrics}
        
        for rule in alert_rules:
            if rule.metric_name not in available_metrics:
                raise ValueError(f"Alert rule references unknown metric: {rule.metric_name}")
        
        return values
    
    def get_metric_by_id(self, metric_id: str) -> Optional[MonitoringMetric]:
        """Retourne une métrique par son ID."""
        for metric in self.metrics + self.custom_metrics:
            if metric.metric_id == metric_id:
                return metric
        return None
    
    def get_dashboard_by_id(self, dashboard_id: str) -> Optional[MonitoringDashboard]:
        """Retourne un dashboard par son ID."""
        for dashboard in self.dashboards:
            if dashboard.dashboard_id == dashboard_id:
                return dashboard
        return None
    
    def get_enabled_metrics(self) -> List[MonitoringMetric]:
        """Retourne les métriques activées."""
        return [m for m in self.metrics + self.custom_metrics if m.enabled]
    
    def get_business_metrics(self) -> List[MonitoringMetric]:
        """Retourne les métriques avec impact business."""
        return [m for m in self.metrics + self.custom_metrics if m.business_impact]
    
    def calculate_storage_requirements(self) -> Dict[str, float]:
        """Calcule les besoins de stockage."""
        total_metrics = len(self.get_enabled_metrics())
        avg_interval = sum(m.sampling_interval_seconds for m in self.get_enabled_metrics()) / max(total_metrics, 1)
        
        # Estimation simple: 8 bytes par point de donnée
        points_per_day = (24 * 3600) / avg_interval
        bytes_per_day = points_per_day * total_metrics * 8
        total_bytes = bytes_per_day * self.retention_days
        
        return {
            "daily_storage_mb": bytes_per_day / (1024 * 1024),
            "total_storage_gb": total_bytes / (1024 * 1024 * 1024),
            "metrics_count": total_metrics,
            "avg_sampling_interval": avg_interval
        }
    
    def export_prometheus_config(self) -> Dict[str, Any]:
        """Exporte la configuration au format Prometheus."""
        return {
            "global": {
                "scrape_interval": f"{self.collection_interval_seconds}s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": target.name,
                    "static_configs": [{"targets": [str(target.endpoint)]}],
                    "scrape_interval": f"{target.scrape_interval_seconds}s",
                    "scrape_timeout": f"{target.timeout_seconds}s",
                    "metrics_path": target.metrics_path
                }
                for target in self.targets if target.enabled
            ],
            "rule_files": ["alerts/*.yml"],
            "alerting": {
                "alertmanagers": [
                    {"static_configs": [{"targets": ["alertmanager:9093"]}]}
                ]
            }
        }
