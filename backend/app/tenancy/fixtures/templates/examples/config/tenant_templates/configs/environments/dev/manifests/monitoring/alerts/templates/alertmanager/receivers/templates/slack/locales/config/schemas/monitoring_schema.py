"""
Schémas de monitoring et alerting - Module Python.

Ce module fournit les classes de validation pour la configuration
du monitoring, des alertes et des tableaux de bord.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class DatasourceType(str, Enum):
    """Types de sources de données."""
    PROMETHEUS = "prometheus"
    ELASTICSEARCH = "elasticsearch"
    INFLUXDB = "influxdb"
    MYSQL = "mysql"
    POSTGRES = "postgres"


class DatabaseType(str, Enum):
    """Types de base de données pour Grafana."""
    SQLITE3 = "sqlite3"
    MYSQL = "mysql"
    POSTGRES = "postgres"


class NotificationChannelType(str, Enum):
    """Types de canaux de notification."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"


class ActionStyle(str, Enum):
    """Styles d'actions Slack."""
    DEFAULT = "default"
    PRIMARY = "primary"
    DANGER = "danger"


class GlobalConfig(BaseModel):
    """Configuration globale du monitoring."""
    scrape_interval: str = Field("15s", regex=r"^[0-9]+[smhd]$")
    evaluation_interval: str = Field("15s", regex=r"^[0-9]+[smhd]$")
    external_labels: Dict[str, str] = Field(default_factory=dict)


class PrometheusConfig(BaseModel):
    """Configuration Prometheus."""
    retention_time: str = Field("15d", regex=r"^[0-9]+[dwmy]$")
    storage_path: str = "/prometheus"
    web_listen_address: str = ":9090"
    web_external_url: Optional[str] = None


class StaticConfig(BaseModel):
    """Configuration statique pour les cibles."""
    targets: List[str]
    labels: Dict[str, str] = Field(default_factory=dict)


class ScrapeConfig(BaseModel):
    """Configuration de collecte des métriques."""
    job_name: str
    scrape_interval: Optional[str] = Field(None, regex=r"^[0-9]+[smhd]$")
    scrape_timeout: Optional[str] = Field(None, regex=r"^[0-9]+[smhd]$")
    metrics_path: str = "/metrics"
    static_configs: List[StaticConfig] = Field(default_factory=list)
    relabel_configs: List[Dict[str, Any]] = Field(default_factory=list)


class PrometheusInstance(BaseModel):
    """Instance Prometheus."""
    enabled: bool = True
    config: PrometheusConfig
    scrape_configs: List[ScrapeConfig] = Field(default_factory=list)


class AlertmanagerGlobalConfig(BaseModel):
    """Configuration globale Alertmanager."""
    smtp_smarthost: Optional[str] = None
    smtp_from: Optional[str] = None
    slack_api_url: Optional[str] = None
    resolve_timeout: str = Field("5m", regex=r"^[0-9]+[smhd]$")


class SlackAction(BaseModel):
    """Action Slack pour les alertes."""
    type: str
    text: str
    url: Optional[str] = None
    style: ActionStyle = ActionStyle.DEFAULT


class SlackField(BaseModel):
    """Champ Slack pour les alertes."""
    title: str
    value: str
    short: bool = False


class SlackConfig(BaseModel):
    """Configuration Slack pour Alertmanager."""
    api_url: Optional[str] = None
    channel: str
    username: str = "alertmanager"
    color: str = "good"
    title: str = "{{ .GroupLabels.alertname }}"
    title_link: Optional[str] = None
    text: str = "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
    fallback: str = "{{ .CommonAnnotations.summary }}"
    pretext: Optional[str] = None
    icon_emoji: str = ":exclamation:"
    icon_url: Optional[str] = None
    link_names: bool = False
    actions: List[SlackAction] = Field(default_factory=list)
    fields: List[SlackField] = Field(default_factory=list)


class EmailConfig(BaseModel):
    """Configuration email pour Alertmanager."""
    to: str
    from_: Optional[str] = Field(None, alias="from")
    subject: str = "{{ .GroupLabels.alertname }}"
    body: str = "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"
    html: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)


class WebhookConfig(BaseModel):
    """Configuration webhook pour Alertmanager."""
    url: str
    send_resolved: bool = True
    http_config: Dict[str, Any] = Field(default_factory=dict)


class Receiver(BaseModel):
    """Récepteur d'alertes."""
    name: str
    email_configs: List[EmailConfig] = Field(default_factory=list)
    slack_configs: List[SlackConfig] = Field(default_factory=list)
    webhook_configs: List[WebhookConfig] = Field(default_factory=list)


class Route(BaseModel):
    """Route d'alertes."""
    group_by: List[str] = Field(default_factory=lambda: ["alertname"])
    group_wait: str = Field("10s", regex=r"^[0-9]+[smhd]$")
    group_interval: str = Field("10s", regex=r"^[0-9]+[smhd]$")
    repeat_interval: str = Field("1h", regex=r"^[0-9]+[smhd]$")
    receiver: str
    routes: List["Route"] = Field(default_factory=list)
    match: Dict[str, str] = Field(default_factory=dict)
    match_re: Dict[str, str] = Field(default_factory=dict)
    continue_: bool = Field(False, alias="continue")


Route.update_forward_refs()


class InhibitRule(BaseModel):
    """Règle d'inhibition d'alertes."""
    source_match: Dict[str, str] = Field(default_factory=dict)
    source_match_re: Dict[str, str] = Field(default_factory=dict)
    target_match: Dict[str, str] = Field(default_factory=dict)
    target_match_re: Dict[str, str] = Field(default_factory=dict)
    equal: List[str] = Field(default_factory=list)


class AlertmanagerConfig(BaseModel):
    """Configuration Alertmanager."""
    global_: AlertmanagerGlobalConfig = Field(alias="global")
    templates: List[str] = Field(default_factory=list)
    route: Route
    receivers: List[Receiver]
    inhibit_rules: List[InhibitRule] = Field(default_factory=list)


class AlertmanagerInstance(BaseModel):
    """Instance Alertmanager."""
    enabled: bool = True
    config: AlertmanagerConfig


class GrafanaServerConfig(BaseModel):
    """Configuration serveur Grafana."""
    http_port: int = Field(3000, ge=1, le=65535)
    domain: str = "localhost"
    root_url: Optional[str] = None


class GrafanaDatabaseConfig(BaseModel):
    """Configuration base de données Grafana."""
    type: DatabaseType = DatabaseType.SQLITE3
    host: Optional[str] = None
    name: str = "grafana"
    user: Optional[str] = None
    password: Optional[str] = None


class GrafanaAuthConfig(BaseModel):
    """Configuration authentification Grafana."""
    disable_login_form: bool = False
    oauth_auto_login: bool = False


class GrafanaSecurityConfig(BaseModel):
    """Configuration sécurité Grafana."""
    admin_user: str = "admin"
    admin_password: str = "admin"
    secret_key: str = "SW2YcwTIb9zpOOhoPsMm"


class GrafanaConfig(BaseModel):
    """Configuration Grafana."""
    server: GrafanaServerConfig
    database: GrafanaDatabaseConfig
    auth: GrafanaAuthConfig
    security: GrafanaSecurityConfig


class Datasource(BaseModel):
    """Source de données Grafana."""
    name: str
    type: DatasourceType
    url: str
    access: str = "proxy"
    isDefault: bool = False


class Dashboard(BaseModel):
    """Tableau de bord Grafana."""
    name: str
    folder: str = "General"
    type: str = "file"
    disableDeletion: bool = False
    updateIntervalSeconds: int = 10
    options: Dict[str, Any] = Field(default_factory=dict)


class GrafanaInstance(BaseModel):
    """Instance Grafana."""
    enabled: bool = True
    config: GrafanaConfig
    datasources: List[Datasource] = Field(default_factory=list)
    dashboards: List[Dashboard] = Field(default_factory=list)


class CustomMetric(BaseModel):
    """Métrique personnalisée tenant."""
    name: str
    query: str
    labels: Dict[str, str] = Field(default_factory=dict)


class CustomAlert(BaseModel):
    """Alerte personnalisée tenant."""
    name: str
    expr: str
    for_: str = Field("5m", alias="for", regex=r"^[0-9]+[smhd]$")
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)


class NotificationChannel(BaseModel):
    """Canal de notification tenant."""
    name: str
    type: NotificationChannelType
    settings: Dict[str, Any] = Field(default_factory=dict)


class TenantSpecificConfig(BaseModel):
    """Configuration spécifique au tenant."""
    tenant_id: str
    custom_metrics: List[CustomMetric] = Field(default_factory=list)
    custom_alerts: List[CustomAlert] = Field(default_factory=list)
    notification_channels: List[NotificationChannel] = Field(default_factory=list)


class MonitoringConfigSchema(BaseModel):
    """Schéma complet de configuration monitoring."""
    global_: GlobalConfig = Field(alias="global")
    prometheus: PrometheusInstance
    alertmanager: AlertmanagerInstance
    grafana: Optional[GrafanaInstance] = None
    tenant_specific: Optional[TenantSpecificConfig] = None

    @validator('prometheus')
    def validate_prometheus(cls, v):
        """Valide la configuration Prometheus."""
        if v.enabled and not v.scrape_configs:
            raise ValueError("Au moins une configuration de collecte doit être définie")
        return v

    @validator('alertmanager')
    def validate_alertmanager(cls, v):
        """Valide la configuration Alertmanager."""
        if v.enabled and not v.config.receivers:
            raise ValueError("Au moins un récepteur doit être configuré")
        return v

    class Config:
        """Configuration Pydantic."""
        use_enum_values = True
        validate_assignment = True
        allow_population_by_field_name = True


class AlertingSchema(BaseModel):
    """Schéma simplifié pour l'alerting."""
    enabled: bool = True
    severity_levels: List[str] = Field(default_factory=lambda: ["critical", "warning", "info"])
    default_receivers: List[str] = Field(default_factory=list)
    escalation_rules: List[Dict[str, Any]] = Field(default_factory=list)
    notification_templates: Dict[str, str] = Field(default_factory=dict)

    class Config:
        """Configuration Pydantic."""
        use_enum_values = True
