"""
Schémas Pydantic avancés pour la configuration de monitoring.

Ce module définit tous les schémas pour Prometheus, Grafana, logging,
tracing et métriques de performance.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, HttpUrl
from decimal import Decimal


class MetricType(str, Enum):
    """Types de métriques Prometheus."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class LogLevel(str, Enum):
    """Niveaux de log."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TracingProvider(str, Enum):
    """Fournisseurs de tracing."""
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"
    OTLP = "otlp"
    DATADOG = "datadog"


class HealthCheckType(str, Enum):
    """Types de health checks."""
    HTTP = "http"
    TCP = "tcp"
    EXEC = "exec"
    GRPC = "grpc"


class StorageRetention(BaseModel):
    """Configuration de rétention du stockage."""
    local: str = Field("15d", description="Rétention locale")
    remote: Optional[str] = Field(None, description="Rétention distante")
    compaction_interval: str = Field("2h", description="Intervalle de compaction")
    retention_size: Optional[str] = Field(None, description="Taille max de rétention")


class ScrapeConfig(BaseModel):
    """Configuration de scraping Prometheus."""
    job_name: str = Field(..., description="Nom du job de scraping")
    scrape_interval: str = Field("30s", description="Intervalle de scraping")
    scrape_timeout: str = Field("10s", description="Timeout de scraping")
    metrics_path: str = Field("/metrics", description="Chemin des métriques")
    scheme: str = Field("http", description="Schéma de connexion")
    static_configs: List[Dict[str, Any]] = Field(default_factory=list, description="Configurations statiques")
    relabel_configs: List[Dict[str, Any]] = Field(default_factory=list, description="Configurations de relabeling")
    metric_relabel_configs: List[Dict[str, Any]] = Field(default_factory=list, description="Relabeling des métriques")
    
    @validator('scrape_interval', 'scrape_timeout')
    def validate_duration(cls, v):
        """Valide le format de durée."""
        import re
        if not re.match(r'^\d+[smh]$', v):
            raise ValueError('Format de durée invalide')
        return v


class PrometheusConfigSchema(BaseModel):
    """Configuration complète de Prometheus."""
    global_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration globale")
    scrape_configs: List[ScrapeConfig] = Field(..., description="Configurations de scraping")
    rule_files: List[str] = Field(default_factory=list, description="Fichiers de règles")
    alerting: Dict[str, Any] = Field(default_factory=dict, description="Configuration d'alerting")
    remote_write: List[Dict[str, Any]] = Field(default_factory=list, description="Écriture distante")
    remote_read: List[Dict[str, Any]] = Field(default_factory=list, description="Lecture distante")
    storage: StorageRetention = Field(default_factory=StorageRetention, description="Configuration de stockage")
    
    @validator('global_config')
    def set_default_global_config(cls, v):
        """Configure les valeurs par défaut globales."""
        defaults = {
            'scrape_interval': '15s',
            'evaluation_interval': '15s',
            'external_labels': {}
        }
        return {**defaults, **v}


class GrafanaDataSource(BaseModel):
    """Configuration d'une source de données Grafana."""
    name: str = Field(..., description="Nom de la source")
    type: str = Field(..., description="Type de source (prometheus, influxdb, etc.)")
    url: HttpUrl = Field(..., description="URL de la source")
    access: str = Field("proxy", description="Mode d'accès")
    is_default: bool = Field(False, description="Source par défaut")
    basic_auth: bool = Field(False, description="Authentification basique")
    basic_auth_user: Optional[str] = Field(None, description="Utilisateur auth basique")
    basic_auth_password: Optional[str] = Field(None, description="Mot de passe auth basique")
    json_data: Dict[str, Any] = Field(default_factory=dict, description="Données JSON")
    secure_json_data: Dict[str, str] = Field(default_factory=dict, description="Données JSON sécurisées")


class GrafanaDashboard(BaseModel):
    """Configuration d'un dashboard Grafana."""
    id: Optional[int] = Field(None, description="ID du dashboard")
    uid: Optional[str] = Field(None, description="UID du dashboard")
    title: str = Field(..., description="Titre du dashboard")
    description: Optional[str] = Field(None, description="Description")
    tags: List[str] = Field(default_factory=list, description="Tags")
    timezone: str = Field("browser", description="Fuseau horaire")
    refresh: str = Field("30s", description="Intervalle de rafraîchissement")
    time_from: str = Field("now-1h", description="Début de la période")
    time_to: str = Field("now", description="Fin de la période")
    panels: List[Dict[str, Any]] = Field(default_factory=list, description="Panneaux du dashboard")
    templating: Dict[str, Any] = Field(default_factory=dict, description="Variables de template")
    annotations: Dict[str, Any] = Field(default_factory=dict, description="Annotations")


class GrafanaConfigSchema(BaseModel):
    """Configuration complète de Grafana."""
    datasources: List[GrafanaDataSource] = Field(..., description="Sources de données")
    dashboards: List[GrafanaDashboard] = Field(default_factory=list, description="Dashboards")
    plugins: List[str] = Field(default_factory=list, description="Plugins installés")
    smtp: Dict[str, Any] = Field(default_factory=dict, description="Configuration SMTP")
    security: Dict[str, Any] = Field(default_factory=dict, description="Configuration sécurité")
    auth: Dict[str, Any] = Field(default_factory=dict, description="Configuration authentification")
    users: Dict[str, Any] = Field(default_factory=dict, description="Configuration utilisateurs")


class LoggingConfigSchema(BaseModel):
    """Configuration du système de logging."""
    level: LogLevel = Field(LogLevel.INFO, description="Niveau de log global")
    format: str = Field("json", description="Format des logs")
    output: str = Field("stdout", description="Sortie des logs")
    rotation: Dict[str, Any] = Field(default_factory=dict, description="Rotation des logs")
    sampling: Dict[str, Any] = Field(default_factory=dict, description="Échantillonnage")
    loggers: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Configuration par logger")
    filters: List[Dict[str, Any]] = Field(default_factory=list, description="Filtres de logs")
    enrichers: List[str] = Field(default_factory=list, description="Enrichisseurs de logs")
    
    class Config:
        use_enum_values = True


class TracingConfigSchema(BaseModel):
    """Configuration du tracing distribué."""
    provider: TracingProvider = Field(TracingProvider.JAEGER, description="Fournisseur de tracing")
    endpoint: HttpUrl = Field(..., description="Endpoint du service de tracing")
    service_name: str = Field(..., description="Nom du service")
    sampling_rate: float = Field(0.1, ge=0.0, le=1.0, description="Taux d'échantillonnage")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags globaux")
    batch_timeout: str = Field("5s", description="Timeout de batch")
    batch_size: int = Field(512, description="Taille de batch")
    max_tag_value_length: int = Field(1024, description="Longueur max des tags")
    
    class Config:
        use_enum_values = True


class HealthCheckConfigSchema(BaseModel):
    """Configuration des health checks."""
    type: HealthCheckType = Field(..., description="Type de health check")
    endpoint: Optional[str] = Field(None, description="Endpoint à vérifier")
    port: Optional[int] = Field(None, description="Port à vérifier")
    path: Optional[str] = Field(None, description="Chemin à vérifier")
    command: Optional[List[str]] = Field(None, description="Commande à exécuter")
    interval: str = Field("30s", description="Intervalle de vérification")
    timeout: str = Field("5s", description="Timeout de vérification")
    retries: int = Field(3, description="Nombre de tentatives")
    initial_delay: str = Field("10s", description="Délai initial")
    success_threshold: int = Field(1, description="Seuil de succès")
    failure_threshold: int = Field(3, description="Seuil d'échec")
    
    class Config:
        use_enum_values = True


class MetricConfigSchema(BaseModel):
    """Configuration d'une métrique."""
    name: str = Field(..., description="Nom de la métrique")
    type: MetricType = Field(..., description="Type de métrique")
    description: str = Field(..., description="Description de la métrique")
    labels: List[str] = Field(default_factory=list, description="Labels de la métrique")
    buckets: Optional[List[float]] = Field(None, description="Buckets pour histogrammes")
    quantiles: Optional[List[float]] = Field(None, description="Quantiles pour summaries")
    unit: Optional[str] = Field(None, description="Unité de mesure")
    
    class Config:
        use_enum_values = True


class PerformanceMetricSchema(BaseModel):
    """Métriques de performance."""
    cpu_usage: MetricConfigSchema = Field(..., description="Utilisation CPU")
    memory_usage: MetricConfigSchema = Field(..., description="Utilisation mémoire")
    disk_usage: MetricConfigSchema = Field(..., description="Utilisation disque")
    network_io: MetricConfigSchema = Field(..., description="I/O réseau")
    response_time: MetricConfigSchema = Field(..., description="Temps de réponse")
    throughput: MetricConfigSchema = Field(..., description="Débit")
    error_rate: MetricConfigSchema = Field(..., description="Taux d'erreur")


class SystemMetricSchema(BaseModel):
    """Métriques système."""
    load_average: MetricConfigSchema = Field(..., description="Charge moyenne")
    file_descriptors: MetricConfigSchema = Field(..., description="Descripteurs de fichiers")
    processes: MetricConfigSchema = Field(..., description="Processus")
    uptime: MetricConfigSchema = Field(..., description="Temps de fonctionnement")
    boot_time: MetricConfigSchema = Field(..., description="Temps de démarrage")


class BusinessMetricSchema(BaseModel):
    """Métriques métier."""
    user_sessions: MetricConfigSchema = Field(..., description="Sessions utilisateur")
    transactions: MetricConfigSchema = Field(..., description="Transactions")
    revenue: MetricConfigSchema = Field(..., description="Revenus")
    conversion_rate: MetricConfigSchema = Field(..., description="Taux de conversion")
    customer_satisfaction: MetricConfigSchema = Field(..., description="Satisfaction client")


class AlertingMetricSchema(BaseModel):
    """Métriques pour les alertes."""
    alerts_fired: MetricConfigSchema = Field(..., description="Alertes déclenchées")
    alerts_resolved: MetricConfigSchema = Field(..., description="Alertes résolues")
    notification_latency: MetricConfigSchema = Field(..., description="Latence de notification")
    escalation_count: MetricConfigSchema = Field(..., description="Nombre d'escalations")


class DashboardConfigSchema(BaseModel):
    """Configuration d'un dashboard de monitoring."""
    name: str = Field(..., description="Nom du dashboard")
    description: Optional[str] = Field(None, description="Description")
    panels: List[Dict[str, Any]] = Field(..., description="Panneaux du dashboard")
    refresh_interval: str = Field("30s", description="Intervalle de rafraîchissement")
    time_range: Dict[str, str] = Field(default_factory=lambda: {"from": "now-1h", "to": "now"})
    variables: List[Dict[str, Any]] = Field(default_factory=list, description="Variables du dashboard")
    annotations: List[Dict[str, Any]] = Field(default_factory=list, description="Annotations")
    tags: List[str] = Field(default_factory=list, description="Tags de classification")


class MonitoringConfigSchema(BaseModel):
    """Configuration complète du monitoring."""
    prometheus: PrometheusConfigSchema = Field(..., description="Configuration Prometheus")
    grafana: GrafanaConfigSchema = Field(..., description="Configuration Grafana")
    logging: LoggingConfigSchema = Field(..., description="Configuration logging")
    tracing: TracingConfigSchema = Field(..., description="Configuration tracing")
    health_checks: List[HealthCheckConfigSchema] = Field(default_factory=list, description="Health checks")
    metrics: Dict[str, MetricConfigSchema] = Field(default_factory=dict, description="Métriques personnalisées")
    performance_metrics: PerformanceMetricSchema = Field(..., description="Métriques de performance")
    system_metrics: SystemMetricSchema = Field(..., description="Métriques système")
    business_metrics: Optional[BusinessMetricSchema] = Field(None, description="Métriques métier")
    alerting_metrics: AlertingMetricSchema = Field(..., description="Métriques d'alerting")
    dashboards: List[DashboardConfigSchema] = Field(default_factory=list, description="Dashboards")
    retention_policies: Dict[str, str] = Field(default_factory=dict, description="Politiques de rétention")
    backup_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration de sauvegarde")
    
    @validator('retention_policies')
    def set_default_retention(cls, v):
        """Configure les politiques de rétention par défaut."""
        defaults = {
            'metrics': '15d',
            'logs': '7d',
            'traces': '3d',
            'alerts': '30d'
        }
        return {**defaults, **v}
