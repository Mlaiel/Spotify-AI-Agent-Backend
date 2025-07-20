"""
Dashboard Schemas - Ultra-Advanced Edition
========================================

Schémas ultra-avancés pour les dashboards interactifs avec widgets dynamiques,
personnalisation avancée et analytics visuels.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Set, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, PositiveInt, NonNegativeFloat, PositiveFloat


class WidgetType(str, Enum):
    """Types de widgets de dashboard."""
    CHART = "chart"
    TABLE = "table"
    KPI = "kpi"
    MAP = "map"
    TEXT = "text"
    IMAGE = "image"
    IFRAME = "iframe"
    CUSTOM = "custom"


class ChartType(str, Enum):
    """Types de graphiques."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    AREA = "area"
    TREEMAP = "treemap"


class DashboardWidget(BaseModel):
    """Widget de dashboard avec configuration avancée."""
    
    widget_id: UUID4 = Field(default_factory=lambda: UUID4())
    title: str = Field(..., description="Titre du widget")
    widget_type: WidgetType = Field(..., description="Type de widget")
    
    # Configuration visuelle
    chart_type: Optional[ChartType] = Field(None, description="Type de graphique")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Configuration")
    
    # Données
    data_source: str = Field(..., description="Source de données")
    query: str = Field(..., description="Requête de données")
    
    # Positionnement
    position_x: int = Field(..., ge=0, description="Position X")
    position_y: int = Field(..., ge=0, description="Position Y")
    width: int = Field(..., gt=0, description="Largeur")
    height: int = Field(..., gt=0, description="Hauteur")
    
    # Interactivité
    interactive: bool = Field(default=True, description="Widget interactif")
    drill_down_enabled: bool = Field(default=False, description="Drill down activé")
    
    # Mise à jour
    auto_refresh: bool = Field(default=True, description="Actualisation auto")
    refresh_interval_seconds: int = Field(default=60, ge=5, description="Intervalle actualisation")


class InteractiveDashboard(BaseModel):
    """Dashboard interactif avec personnalisation avancée."""
    
    dashboard_id: UUID4 = Field(default_factory=lambda: UUID4())
    name: str = Field(..., description="Nom du dashboard")
    description: Optional[str] = Field(None, description="Description")
    
    # Propriétaire et partage
    owner_id: UUID4 = Field(..., description="Propriétaire")
    tenant_id: UUID4 = Field(..., description="Tenant")
    is_public: bool = Field(default=False, description="Dashboard public")
    shared_with: List[UUID4] = Field(default_factory=list, description="Partagé avec")
    
    # Configuration
    layout_type: str = Field(default="grid", description="Type de layout")
    theme: str = Field(default="default", description="Thème")
    
    # Widgets
    widgets: List[DashboardWidget] = Field(default_factory=list, description="Widgets")
    
    # Métadonnées
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_viewed_at: Optional[datetime] = Field(None, description="Dernière consultation")
    view_count: int = Field(default=0, ge=0, description="Nombre de vues")
    
    # Performance
    load_time_ms: Optional[float] = Field(None, description="Temps de chargement")
    data_freshness: Optional[datetime] = Field(None, description="Fraîcheur des données")


# Export
__all__ = ["WidgetType", "ChartType", "DashboardWidget", "InteractiveDashboard"]
