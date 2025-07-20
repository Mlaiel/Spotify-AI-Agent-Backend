"""
Advanced Dashboard Schemas - Industrial Grade Visualization System
=================================================================

Ce module définit des schémas de tableaux de bord ultra-avancés pour monitoring
industriel avec intelligence artificielle, visualisations interactives et
analytics en temps réel.

Features:
- Multi-dimensional interactive dashboards
- Real-time streaming data visualization
- ML-powered anomaly highlighting
- Executive and operational views
- Mobile-responsive layouts
- Custom widgets and plugins
- Drill-down and correlation views
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import json
from .metric_schemas import MetricCategory, MetricType
from .alert_schemas import AlertPriority


class DashboardType(str, Enum):
    """Types de tableaux de bord"""
    EXECUTIVE = "executive"              # Vue dirigeants/C-level
    OPERATIONAL = "operational"          # Vue opérationnelle
    TECHNICAL = "technical"              # Vue technique détaillée
    SECURITY = "security"                # Vue sécurité
    BUSINESS = "business"                # Vue business/KPIs
    ML_AI = "ml_ai"                     # Vue ML/IA
    FINANCIAL = "financial"              # Vue financière/coûts
    COMPLIANCE = "compliance"            # Vue conformité
    CUSTOM = "custom"                    # Vue personnalisée


class VisualizationType(str, Enum):
    """Types de visualisation"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    GAUGE = "gauge"
    SINGLE_STAT = "single_stat"
    TABLE = "table"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    SCATTER_PLOT = "scatter_plot"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    FLAME_GRAPH = "flame_graph"
    TOPOLOGY = "topology"
    GRAPH = "graph"
    MAP = "map"
    TIMELINE = "timeline"
    GANTT = "gantt"
    CANDLESTICK = "candlestick"
    RADAR = "radar"
    SUNBURST = "sunburst"
    WATERFALL = "waterfall"


class RefreshInterval(str, Enum):
    """Intervalles de rafraîchissement"""
    REALTIME = "realtime"        # Streaming temps réel
    FIVE_SECONDS = "5s"
    TEN_SECONDS = "10s"
    THIRTY_SECONDS = "30s"
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    SIX_HOURS = "6h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"


class TimeRange(str, Enum):
    """Plages temporelles"""
    LAST_5_MINUTES = "5m"
    LAST_15_MINUTES = "15m"
    LAST_30_MINUTES = "30m"
    LAST_1_HOUR = "1h"
    LAST_3_HOURS = "3h"
    LAST_6_HOURS = "6h"
    LAST_12_HOURS = "12h"
    LAST_24_HOURS = "24h"
    LAST_2_DAYS = "2d"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    LAST_90_DAYS = "90d"
    CUSTOM = "custom"


class WidgetSize(str, Enum):
    """Tailles de widgets"""
    SMALL = "small"          # 1x1
    MEDIUM = "medium"        # 2x2
    LARGE = "large"          # 3x2
    XLARGE = "xlarge"        # 4x3
    FULL_WIDTH = "full_width" # 12x2
    CUSTOM = "custom"


class ThresholdDisplay(BaseModel):
    """Configuration d'affichage des seuils"""
    show_warning: bool = Field(True, description="Afficher seuil warning")
    show_critical: bool = Field(True, description="Afficher seuil critique")
    warning_color: str = Field("#ff9800", description="Couleur warning")
    critical_color: str = Field("#f44336", description="Couleur critique")
    good_color: str = Field("#4caf50", description="Couleur OK")


class MLVisualizationConfig(BaseModel):
    """Configuration visualisation ML"""
    show_anomalies: bool = Field(True, description="Highlighting anomalies")
    show_predictions: bool = Field(True, description="Afficher prédictions")
    show_confidence_bands: bool = Field(True, description="Bandes de confiance")
    anomaly_color: str = Field("#e91e63", description="Couleur anomalies")
    prediction_color: str = Field("#2196f3", description="Couleur prédictions")
    confidence_alpha: float = Field(0.3, description="Transparence bandes")
    
    @validator('confidence_alpha')
    def validate_alpha(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Alpha must be between 0 and 1')
        return v


class DrillDownConfig(BaseModel):
    """Configuration drill-down"""
    enabled: bool = Field(True, description="Drill-down activé")
    target_dashboard: Optional[str] = Field(None, description="Dashboard cible")
    target_panel: Optional[str] = Field(None, description="Panel cible")
    filter_params: Dict[str, str] = Field(default_factory=dict, description="Paramètres de filtre")
    preserve_time_range: bool = Field(True, description="Conserver plage temporelle")


class AlertIntegration(BaseModel):
    """Intégration avec les alertes"""
    show_alert_annotations: bool = Field(True, description="Annotations d'alertes")
    highlight_firing_alerts: bool = Field(True, description="Surligner alertes actives")
    alert_color_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "critical": "#f44336",
            "high": "#ff9800", 
            "medium": "#ffeb3b",
            "low": "#4caf50"
        },
        description="Mapping couleurs alertes"
    )


class Widget(BaseModel):
    """Widget de tableau de bord avancé"""
    
    # Identifiants
    id: str = Field(..., description="ID unique du widget")
    title: str = Field(..., description="Titre du widget")
    description: Optional[str] = Field(None, description="Description")
    
    # Type et visualisation
    visualization_type: VisualizationType = Field(..., description="Type de visualisation")
    category: MetricCategory = Field(..., description="Catégorie")
    
    # Données et requêtes
    query: str = Field(..., description="Requête de données")
    metrics: List[str] = Field(default_factory=list, description="Métriques utilisées")
    
    # Configuration temporelle
    time_range: TimeRange = Field(TimeRange.LAST_1_HOUR, description="Plage temporelle")
    refresh_interval: RefreshInterval = Field(RefreshInterval.ONE_MINUTE, description="Intervalle refresh")
    
    # Positionnement et taille
    position: Dict[str, int] = Field(..., description="Position (x, y)")
    size: WidgetSize = Field(WidgetSize.MEDIUM, description="Taille")
    custom_size: Optional[Dict[str, int]] = Field(None, description="Taille personnalisée")
    
    # Configuration visuelle
    color_scheme: str = Field("default", description="Schéma de couleurs")
    show_legend: bool = Field(True, description="Afficher légende")
    show_grid: bool = Field(True, description="Afficher grille")
    
    # Seuils et alertes
    thresholds: Optional[ThresholdDisplay] = Field(None, description="Configuration seuils")
    alert_integration: AlertIntegration = Field(default_factory=AlertIntegration)
    
    # ML et IA
    ml_config: Optional[MLVisualizationConfig] = Field(None, description="Configuration ML")
    
    # Interaction
    drill_down: Optional[DrillDownConfig] = Field(None, description="Configuration drill-down")
    tooltip_template: Optional[str] = Field(None, description="Template tooltip")
    
    # Formatage
    unit: str = Field("", description="Unité d'affichage")
    decimal_places: int = Field(2, description="Nombre de décimales")
    format_type: str = Field("number", description="Type de formatage")
    
    # Options spécifiques au type
    visualization_options: Dict[str, Any] = Field(
        default_factory=dict, description="Options spécifiques visualisation"
    )
    
    # Métadonnées
    tags: List[str] = Field(default_factory=list, description="Tags")
    owner: str = Field("", description="Propriétaire")
    
    # État
    enabled: bool = Field(True, description="Widget activé")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "spotify_api_latency_widget",
                "title": "Spotify API Latency P95",
                "visualization_type": "line_chart",
                "query": "histogram_quantile(0.95, spotify_api_response_time)",
                "position": {"x": 0, "y": 0},
                "size": "large",
                "ml_config": {
                    "show_anomalies": True,
                    "show_predictions": True
                },
                "thresholds": {
                    "show_warning": True,
                    "show_critical": True
                }
            }
        }


class DashboardLayout(BaseModel):
    """Configuration de layout de tableau de bord"""
    grid_size: Dict[str, int] = Field(
        default_factory=lambda: {"columns": 12, "rows": 20},
        description="Taille de la grille"
    )
    responsive: bool = Field(True, description="Layout responsive")
    mobile_layout: Optional[Dict[str, Any]] = Field(None, description="Layout mobile")
    tablet_layout: Optional[Dict[str, Any]] = Field(None, description="Layout tablette")


class DashboardFilter(BaseModel):
    """Filtre de tableau de bord"""
    name: str = Field(..., description="Nom du filtre")
    label: str = Field(..., description="Label d'affichage")
    type: str = Field("select", description="Type de filtre")
    
    # Options
    options: List[Dict[str, str]] = Field(default_factory=list, description="Options")
    default_value: Optional[str] = Field(None, description="Valeur par défaut")
    
    # Configuration
    multi_select: bool = Field(False, description="Sélection multiple")
    searchable: bool = Field(True, description="Recherche activée")
    
    # Variable template
    variable_name: str = Field(..., description="Nom de variable template")


class DashboardVariable(BaseModel):
    """Variable de tableau de bord"""
    name: str = Field(..., description="Nom de la variable")
    label: str = Field(..., description="Label d'affichage")
    type: str = Field("query", description="Type de variable")
    
    # Requête pour valeurs
    query: str = Field(..., description="Requête pour récupérer valeurs")
    refresh: str = Field("on_dashboard_load", description="Quand rafraîchir")
    
    # Options
    multi_value: bool = Field(False, description="Valeurs multiples")
    include_all: bool = Field(False, description="Inclure option 'All'")
    
    # Valeurs
    current_value: Optional[str] = Field(None, description="Valeur actuelle")
    default_value: Optional[str] = Field(None, description="Valeur par défaut")


class DashboardAnnotation(BaseModel):
    """Annotation de tableau de bord"""
    name: str = Field(..., description="Nom de l'annotation")
    query: str = Field(..., description="Requête pour annotations")
    
    # Style
    color: str = Field("#ff0000", description="Couleur")
    icon: str = Field("bolt", description="Icône")
    
    # Affichage
    show_in_legend: bool = Field(True, description="Afficher dans légende")
    text_template: str = Field("{{title}}", description="Template texte")


class Dashboard(BaseModel):
    """Tableau de bord ultra-avancé"""
    
    # Identifiants
    id: str = Field(..., description="ID unique du dashboard")
    title: str = Field(..., description="Titre du dashboard")
    description: str = Field("", description="Description")
    
    # Type et catégorie
    dashboard_type: DashboardType = Field(..., description="Type de dashboard")
    category: MetricCategory = Field(..., description="Catégorie principale")
    
    # Widgets
    widgets: List[Widget] = Field(default_factory=list, description="Widgets du dashboard")
    
    # Layout et responsive
    layout: DashboardLayout = Field(default_factory=DashboardLayout, description="Configuration layout")
    
    # Temporel
    default_time_range: TimeRange = Field(TimeRange.LAST_1_HOUR, description="Plage temporelle par défaut")
    default_refresh_interval: RefreshInterval = Field(
        RefreshInterval.ONE_MINUTE, description="Intervalle refresh par défaut"
    )
    auto_refresh: bool = Field(True, description="Auto-refresh activé")
    
    # Filtres et variables
    filters: List[DashboardFilter] = Field(default_factory=list, description="Filtres")
    variables: List[DashboardVariable] = Field(default_factory=list, description="Variables")
    
    # Annotations
    annotations: List[DashboardAnnotation] = Field(default_factory=list, description="Annotations")
    
    # Thème et style
    theme: str = Field("dark", description="Thème (dark, light)")
    color_palette: str = Field("default", description="Palette de couleurs")
    
    # Permissions et accès
    public: bool = Field(False, description="Dashboard public")
    allowed_users: List[str] = Field(default_factory=list, description="Utilisateurs autorisés")
    allowed_teams: List[str] = Field(default_factory=list, description="Équipes autorisées")
    
    # Alertes intégrées
    show_alert_list: bool = Field(True, description="Afficher liste alertes")
    alert_severity_filter: List[str] = Field(
        default_factory=lambda: ["critical", "high"],
        description="Filtres sévérité alertes"
    )
    
    # ML et IA
    ml_insights_enabled: bool = Field(True, description="Insights ML activés")
    anomaly_detection_enabled: bool = Field(True, description="Détection anomalies")
    predictive_analytics: bool = Field(False, description="Analytics prédictifs")
    
    # Export et partage
    exportable: bool = Field(True, description="Exportable")
    shareable: bool = Field(True, description="Partageable")
    
    # Métadonnées
    tags: List[str] = Field(default_factory=list, description="Tags")
    owner: str = Field("", description="Propriétaire")
    team: str = Field("", description="Équipe")
    environment: str = Field("production", description="Environnement")
    
    # Audit
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field("1.0.0", description="Version")
    
    # État
    enabled: bool = Field(True, description="Dashboard activé")
    favorite: bool = Field(False, description="Dashboard favori")
    
    def add_widget(self, widget: Widget) -> None:
        """Ajouter un widget au dashboard"""
        if any(w.id == widget.id for w in self.widgets):
            raise ValueError(f"Widget {widget.id} already exists")
        self.widgets.append(widget)
    
    def get_widget(self, widget_id: str) -> Optional[Widget]:
        """Récupérer un widget par ID"""
        return next((w for w in self.widgets if w.id == widget_id), None)
    
    def remove_widget(self, widget_id: str) -> bool:
        """Supprimer un widget"""
        widget = self.get_widget(widget_id)
        if widget:
            self.widgets.remove(widget)
            return True
        return False
    
    class Config:
        schema_extra = {
            "example": {
                "id": "spotify_executive_dashboard",
                "title": "Spotify AI Agent - Executive Overview",
                "dashboard_type": "executive",
                "category": "business",
                "description": "Vue exécutive des KPIs business et techniques",
                "default_time_range": "24h",
                "ml_insights_enabled": True,
                "predictive_analytics": True
            }
        }


class DashboardRegistry(BaseModel):
    """Registre centralisé des tableaux de bord"""
    
    dashboards: List[Dashboard] = Field(default_factory=list, description="Dashboards")
    
    # Organisation
    categories: Dict[str, List[str]] = Field(default_factory=dict, description="Catégories")
    folders: Dict[str, List[str]] = Field(default_factory=dict, description="Dossiers")
    
    # Configuration globale
    global_theme: str = Field("dark", description="Thème global")
    global_refresh_interval: RefreshInterval = Field(
        RefreshInterval.ONE_MINUTE, description="Intervalle global"
    )
    
    def add_dashboard(self, dashboard: Dashboard) -> None:
        """Ajouter un dashboard"""
        if any(d.id == dashboard.id for d in self.dashboards):
            raise ValueError(f"Dashboard {dashboard.id} already exists")
        self.dashboards.append(dashboard)
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Récupérer un dashboard par ID"""
        return next((d for d in self.dashboards if d.id == dashboard_id), None)
    
    def get_dashboards_by_type(self, dashboard_type: DashboardType) -> List[Dashboard]:
        """Récupérer dashboards par type"""
        return [d for d in self.dashboards if d.dashboard_type == dashboard_type]
    
    def get_dashboards_by_category(self, category: MetricCategory) -> List[Dashboard]:
        """Récupérer dashboards par catégorie"""
        return [d for d in self.dashboards if d.category == category]


# Tableaux de bord prédéfinis pour Spotify AI Agent
def create_executive_dashboard() -> Dashboard:
    """Créer dashboard exécutif"""
    dashboard = Dashboard(
        id="spotify_executive_overview",
        title="Spotify AI Agent - Executive Overview",
        description="Vue exécutive des KPIs business et performance globale",
        dashboard_type=DashboardType.EXECUTIVE,
        category=MetricCategory.BUSINESS,
        default_time_range=TimeRange.LAST_24_HOURS,
        ml_insights_enabled=True,
        predictive_analytics=True
    )
    
    # Widget KPI Revenue
    revenue_widget = Widget(
        id="revenue_kpi",
        title="Revenue Impact",
        visualization_type=VisualizationType.SINGLE_STAT,
        category=MetricCategory.FINANCIAL,
        query="sum(spotify_revenue_impact)",
        position={"x": 0, "y": 0},
        size=WidgetSize.MEDIUM,
        format_type="currency"
    )
    
    # Widget User Engagement
    engagement_widget = Widget(
        id="user_engagement",
        title="User Engagement Score",
        visualization_type=VisualizationType.GAUGE,
        category=MetricCategory.BUSINESS,
        query="avg(spotify_user_engagement_score)",
        position={"x": 2, "y": 0},
        size=WidgetSize.MEDIUM,
        ml_config=MLVisualizationConfig(show_anomalies=True, show_predictions=True)
    )
    
    # Widget System Health
    health_widget = Widget(
        id="system_health",
        title="System Health Overview",
        visualization_type=VisualizationType.PIE_CHART,
        category=MetricCategory.INFRASTRUCTURE,
        query="spotify_system_health_status",
        position={"x": 4, "y": 0},
        size=WidgetSize.MEDIUM
    )
    
    dashboard.add_widget(revenue_widget)
    dashboard.add_widget(engagement_widget) 
    dashboard.add_widget(health_widget)
    
    return dashboard


def create_operational_dashboard() -> Dashboard:
    """Créer dashboard opérationnel"""
    dashboard = Dashboard(
        id="spotify_operational_monitoring",
        title="Spotify AI Agent - Operational Monitoring",
        description="Monitoring opérationnel temps réel avec alertes",
        dashboard_type=DashboardType.OPERATIONAL,
        category=MetricCategory.APPLICATION,
        default_time_range=TimeRange.LAST_1_HOUR,
        default_refresh_interval=RefreshInterval.THIRTY_SECONDS,
        anomaly_detection_enabled=True
    )
    
    # Widget API Latency
    latency_widget = Widget(
        id="api_latency_p95",
        title="API Latency P95",
        visualization_type=VisualizationType.LINE_CHART,
        category=MetricCategory.APPLICATION,
        query="histogram_quantile(0.95, spotify_api_response_time)",
        position={"x": 0, "y": 0},
        size=WidgetSize.LARGE,
        ml_config=MLVisualizationConfig(show_anomalies=True),
        thresholds=ThresholdDisplay()
    )
    
    # Widget Error Rate
    error_widget = Widget(
        id="error_rate",
        title="Error Rate",
        visualization_type=VisualizationType.LINE_CHART,
        category=MetricCategory.APPLICATION,
        query="rate(spotify_api_errors_total[5m])",
        position={"x": 3, "y": 0},
        size=WidgetSize.LARGE,
        alert_integration=AlertIntegration(highlight_firing_alerts=True)
    )
    
    dashboard.add_widget(latency_widget)
    dashboard.add_widget(error_widget)
    
    return dashboard


def create_ml_dashboard() -> Dashboard:
    """Créer dashboard ML/IA"""
    dashboard = Dashboard(
        id="spotify_ml_ai_monitoring",
        title="Spotify AI Agent - ML/AI Monitoring",
        description="Monitoring des modèles ML et performances IA",
        dashboard_type=DashboardType.ML_AI,
        category=MetricCategory.ML_AI,
        ml_insights_enabled=True,
        predictive_analytics=True
    )
    
    # Widget Model Accuracy
    accuracy_widget = Widget(
        id="model_accuracy_trend",
        title="Model Accuracy Trends",
        visualization_type=VisualizationType.LINE_CHART,
        category=MetricCategory.ML_AI,
        query="spotify_model_accuracy",
        position={"x": 0, "y": 0},
        size=WidgetSize.XLARGE,
        ml_config=MLVisualizationConfig(
            show_anomalies=True,
            show_predictions=True,
            show_confidence_bands=True
        )
    )
    
    dashboard.add_widget(accuracy_widget)
    
    return dashboard


def create_default_dashboard_registry() -> DashboardRegistry:
    """Créer registre avec dashboards par défaut"""
    registry = DashboardRegistry()
    
    # Ajouter les dashboards prédéfinis
    registry.add_dashboard(create_executive_dashboard())
    registry.add_dashboard(create_operational_dashboard())
    registry.add_dashboard(create_ml_dashboard())
    
    return registry


# Export des classes principales
__all__ = [
    "DashboardType",
    "VisualizationType", 
    "RefreshInterval",
    "TimeRange",
    "WidgetSize",
    "ThresholdDisplay",
    "MLVisualizationConfig",
    "DrillDownConfig",
    "AlertIntegration",
    "Widget",
    "DashboardLayout",
    "DashboardFilter",
    "DashboardVariable",
    "DashboardAnnotation",
    "Dashboard",
    "DashboardRegistry",
    "create_executive_dashboard",
    "create_operational_dashboard",
    "create_ml_dashboard",
    "create_default_dashboard_registry"
]
