"""
Schémas d'analytiques d'alertes - Spotify AI Agent
Analyse avancée et reporting des données d'alertes
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from uuid import UUID, uuid4
from enum import Enum
import json
from statistics import mean, median, stdev
from decimal import Decimal

from pydantic import BaseModel, Field, validator, computed_field, ConfigDict

from . import (
    BaseSchema, TimestampMixin, TenantMixin, MetadataMixin,
    AlertLevel, AlertStatus, WarningCategory, Priority, Environment
)


class AnalyticsTimeframe(str, Enum):
    """Périodes d'analyse"""
    LAST_HOUR = "last_hour"
    LAST_24_HOURS = "last_24_hours"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"


class MetricAggregation(str, Enum):
    """Types d'agrégation de métriques"""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    STANDARD_DEVIATION = "stddev"
    RATE = "rate"


class TrendDirection(str, Enum):
    """Direction de tendance"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class ReportFormat(str, Enum):
    """Formats de rapport"""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    HTML = "html"
    XLSX = "xlsx"
    PNG = "png"  # Graphiques


class PerformanceMetric(BaseModel):
    """Métrique de performance"""
    
    name: str = Field(..., min_length=1, max_length=100)
    value: Union[int, float, Decimal] = Field(...)
    unit: Optional[str] = Field(None, max_length=20)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Contexte de la métrique
    category: Optional[str] = Field(None, max_length=50)
    source: Optional[str] = Field(None, max_length=100)
    tags: Dict[str, str] = Field(default_factory=dict)
    
    # Comparaison
    previous_value: Optional[Union[int, float, Decimal]] = Field(None)
    change_percentage: Optional[float] = Field(None)
    
    @computed_field
    @property
    def change_direction(self) -> Optional[TrendDirection]:
        """Direction du changement"""
        if self.previous_value is None:
            return None
        
        current = float(self.value)
        previous = float(self.previous_value)
        
        if abs(current - previous) < 0.01:  # Seuil de stabilité
            return TrendDirection.STABLE
        elif current > previous:
            return TrendDirection.INCREASING
        else:
            return TrendDirection.DECREASING


class AlertAnalytics(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Analytiques d'alertes"""
    
    analytics_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Période d'analyse
    timeframe: AnalyticsTimeframe = Field(...)
    start_time: Optional[datetime] = Field(None)
    end_time: Optional[datetime] = Field(None)
    
    # Filtres d'analyse
    environment_filter: List[Environment] = Field(default_factory=list)
    severity_filter: List[AlertLevel] = Field(default_factory=list)
    category_filter: List[WarningCategory] = Field(default_factory=list)
    
    # Métriques de base
    total_alerts: int = Field(0, ge=0)
    active_alerts: int = Field(0, ge=0)
    resolved_alerts: int = Field(0, ge=0)
    suppressed_alerts: int = Field(0, ge=0)
    
    # Métriques de performance
    avg_resolution_time_minutes: Optional[float] = Field(None, ge=0)
    median_resolution_time_minutes: Optional[float] = Field(None, ge=0)
    mttr_minutes: Optional[float] = Field(None, ge=0)  # Mean Time To Resolution
    mtbf_hours: Optional[float] = Field(None, ge=0)   # Mean Time Between Failures
    
    # Distribution par sévérité
    severity_distribution: Dict[str, int] = Field(default_factory=dict)
    severity_trends: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    
    # Distribution par catégorie
    category_distribution: Dict[str, int] = Field(default_factory=dict)
    category_trends: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    
    # Top alertes
    top_alert_sources: List[Dict[str, Any]] = Field(default_factory=list)
    top_noisy_rules: List[Dict[str, Any]] = Field(default_factory=list)
    most_frequent_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Métriques de corrélation
    correlation_rate: Optional[float] = Field(None, ge=0, le=100)
    avg_correlation_size: Optional[float] = Field(None, ge=0)
    correlation_effectiveness: Optional[float] = Field(None, ge=0, le=100)
    
    # Métriques de notification
    notification_success_rate: Optional[float] = Field(None, ge=0, le=100)
    avg_notification_delay_seconds: Optional[float] = Field(None, ge=0)
    notification_channel_usage: Dict[str, int] = Field(default_factory=dict)
    
    # Insights et recommandations
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    anomalies_detected: List[Dict[str, Any]] = Field(default_factory=list)
    
    # État de l'analyse
    analysis_completed: bool = Field(False)
    analysis_duration_seconds: Optional[float] = Field(None, ge=0)
    analysis_errors: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )

    @computed_field
    @property
    def alert_rate_per_hour(self) -> Optional[float]:
        """Taux d'alertes par heure"""
        if not self.start_time or not self.end_time:
            return None
        
        duration_hours = (self.end_time - self.start_time).total_seconds() / 3600
        if duration_hours <= 0:
            return None
        
        return self.total_alerts / duration_hours

    @computed_field
    @property
    def resolution_rate(self) -> float:
        """Taux de résolution des alertes"""
        if self.total_alerts == 0:
            return 0.0
        return (self.resolved_alerts / self.total_alerts) * 100

    @computed_field
    @property
    def critical_alert_percentage(self) -> float:
        """Pourcentage d'alertes critiques"""
        if self.total_alerts == 0:
            return 0.0
        
        critical_count = self.severity_distribution.get('critical', 0)
        return (critical_count / self.total_alerts) * 100

    def calculate_trends(self) -> Dict[str, TrendDirection]:
        """Calcule les tendances pour différentes métriques"""
        trends = {}
        
        # Analyser les tendances de sévérité
        for severity, trend_data in self.severity_trends.items():
            if len(trend_data) >= 2:
                values = [point['value'] for point in trend_data]
                if len(values) >= 2:
                    recent_avg = mean(values[-3:]) if len(values) >= 3 else values[-1]
                    older_avg = mean(values[:3]) if len(values) >= 3 else values[0]
                    
                    if abs(recent_avg - older_avg) < 0.1:
                        trends[f'severity_{severity}'] = TrendDirection.STABLE
                    elif recent_avg > older_avg:
                        trends[f'severity_{severity}'] = TrendDirection.INCREASING
                    else:
                        trends[f'severity_{severity}'] = TrendDirection.DECREASING
        
        return trends

    def generate_insights(self) -> List[str]:
        """Génère des insights automatiques"""
        insights = []
        
        # Insight sur le taux de résolution
        if self.resolution_rate < 80:
            insights.append(
                f"Taux de résolution faible ({self.resolution_rate:.1f}%). "
                "Considérez l'optimisation des processus de résolution."
            )
        
        # Insight sur les alertes critiques
        if self.critical_alert_percentage > 30:
            insights.append(
                f"Proportion élevée d'alertes critiques ({self.critical_alert_percentage:.1f}%). "
                "Vérifiez les seuils d'alerte et les règles de classification."
            )
        
        # Insight sur le MTTR
        if self.mttr_minutes and self.mttr_minutes > 60:
            insights.append(
                f"MTTR élevé ({self.mttr_minutes:.1f} minutes). "
                "Optimisez les processus de détection et de réponse."
            )
        
        # Insight sur les sources bruyantes
        if self.top_noisy_rules:
            noisiest = self.top_noisy_rules[0]
            if noisiest.get('count', 0) > self.total_alerts * 0.2:
                insights.append(
                    f"Règle très bruyante détectée: {noisiest.get('name')}. "
                    "Considérez l'ajustement des seuils ou la suppression."
                )
        
        return insights

    def generate_recommendations(self) -> List[str]:
        """Génère des recommandations automatiques"""
        recommendations = []
        
        # Recommandations basées sur les métriques
        if self.correlation_rate and self.correlation_rate < 20:
            recommendations.append(
                "Activez ou optimisez la corrélation d'alertes pour réduire le bruit."
            )
        
        if self.notification_success_rate and self.notification_success_rate < 95:
            recommendations.append(
                "Vérifiez la configuration des canaux de notification et leur santé."
            )
        
        if self.avg_notification_delay_seconds and self.avg_notification_delay_seconds > 60:
            recommendations.append(
                "Optimisez les délais de notification pour améliorer la réactivité."
            )
        
        # Recommandations basées sur les tendances
        trends = self.calculate_trends()
        for metric, direction in trends.items():
            if direction == TrendDirection.INCREASING and 'critical' in metric:
                recommendations.append(
                    "Tendance à la hausse des alertes critiques détectée. "
                    "Analysez les causes racines et renforcez la surveillance préventive."
                )
        
        return recommendations


class TrendAnalysis(BaseSchema, TimestampMixin, TenantMixin):
    """Analyse de tendances d'alertes"""
    
    analysis_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    
    # Configuration
    metric_name: str = Field(..., min_length=1, max_length=100)
    aggregation: MetricAggregation = Field(...)
    window_size_minutes: int = Field(60, ge=1, le=10080)
    
    # Données de tendance
    data_points: List[Dict[str, Any]] = Field(default_factory=list)
    trend_direction: TrendDirection = Field(TrendDirection.UNKNOWN)
    trend_strength: float = Field(0.0, ge=0.0, le=1.0)
    
    # Statistiques
    mean_value: Optional[float] = Field(None)
    median_value: Optional[float] = Field(None)
    std_deviation: Optional[float] = Field(None)
    min_value: Optional[float] = Field(None)
    max_value: Optional[float] = Field(None)
    
    # Détection d'anomalies
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    seasonal_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Prédictions
    predicted_values: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_interval: Optional[Tuple[float, float]] = Field(None)
    
    @computed_field
    @property
    def data_point_count(self) -> int:
        """Nombre de points de données"""
        return len(self.data_points)

    @computed_field
    @property
    def trend_duration_hours(self) -> Optional[float]:
        """Durée de la tendance en heures"""
        if len(self.data_points) < 2:
            return None
        
        start_time = self.data_points[0]['timestamp']
        end_time = self.data_points[-1]['timestamp']
        
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        duration = end_time - start_time
        return duration.total_seconds() / 3600

    def calculate_trend(self):
        """Calcule la tendance et les statistiques"""
        if len(self.data_points) < 2:
            return
        
        values = [point['value'] for point in self.data_points]
        
        # Statistiques de base
        self.mean_value = mean(values)
        self.median_value = median(values)
        self.min_value = min(values)
        self.max_value = max(values)
        
        if len(values) > 1:
            self.std_deviation = stdev(values)
        
        # Calcul de la tendance (régression linéaire simple)
        n = len(values)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        # Pente de la droite de régression
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Déterminer la direction et la force de la tendance
        if abs(slope) < 0.01:
            self.trend_direction = TrendDirection.STABLE
            self.trend_strength = 0.0
        elif slope > 0:
            self.trend_direction = TrendDirection.INCREASING
            self.trend_strength = min(abs(slope) / (self.max_value - self.min_value), 1.0)
        else:
            self.trend_direction = TrendDirection.DECREASING
            self.trend_strength = min(abs(slope) / (self.max_value - self.min_value), 1.0)

    def detect_anomalies(self, threshold_std: float = 2.0):
        """Détecte les anomalies dans les données"""
        if not self.mean_value or not self.std_deviation:
            return
        
        self.anomalies = []
        
        for i, point in enumerate(self.data_points):
            value = point['value']
            z_score = abs(value - self.mean_value) / self.std_deviation
            
            if z_score > threshold_std:
                self.anomalies.append({
                    'index': i,
                    'timestamp': point['timestamp'],
                    'value': value,
                    'z_score': z_score,
                    'type': 'outlier'
                })


class AlertReport(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Rapport d'alertes"""
    
    report_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Configuration du rapport
    report_type: str = Field(..., min_length=1, max_length=100)
    format: ReportFormat = Field(...)
    schedule: Optional[str] = Field(None)  # Expression cron
    
    # Période couverte
    start_time: datetime = Field(...)
    end_time: datetime = Field(...)
    
    # Contenu du rapport
    analytics_ids: List[UUID] = Field(default_factory=list)
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Métriques incluses
    included_metrics: List[str] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Destinataires
    recipients: List[str] = Field(default_factory=list)  # Emails ou IDs
    distribution_channels: List[str] = Field(default_factory=list)
    
    # État de génération
    status: str = Field("draft")  # draft, generating, completed, failed
    generated_at: Optional[datetime] = Field(None)
    file_path: Optional[str] = Field(None)
    file_size_bytes: Optional[int] = Field(None, ge=0)
    
    # Configuration avancée
    auto_generated: bool = Field(False)
    retention_days: int = Field(30, ge=1, le=365)
    confidentiality_level: str = Field("internal")
    
    @computed_field
    @property
    def report_duration_hours(self) -> float:
        """Durée couverte par le rapport en heures"""
        duration = self.end_time - self.start_time
        return duration.total_seconds() / 3600

    def add_section(self, title: str, content: Any, section_type: str = "text"):
        """Ajoute une section au rapport"""
        section = {
            'title': title,
            'type': section_type,
            'content': content,
            'order': len(self.sections) + 1
        }
        self.sections.append(section)

    def add_chart(self, title: str, chart_type: str, data: Dict[str, Any]):
        """Ajoute un graphique au rapport"""
        chart = {
            'title': title,
            'type': chart_type,
            'data': data,
            'id': f"chart_{len(self.charts) + 1}"
        }
        self.charts.append(chart)


class DashboardConfig(BaseSchema, TimestampMixin, TenantMixin, MetadataMixin):
    """Configuration de tableau de bord d'alertes"""
    
    dashboard_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Layout et widgets
    layout: Dict[str, Any] = Field(default_factory=dict)
    widgets: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Filtres par défaut
    default_timeframe: AnalyticsTimeframe = Field(AnalyticsTimeframe.LAST_24_HOURS)
    default_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Rafraîchissement
    auto_refresh_interval_seconds: int = Field(300, ge=30, le=3600)
    real_time_updates: bool = Field(False)
    
    # Permissions et partage
    public: bool = Field(False)
    shared_with: List[UUID] = Field(default_factory=list)
    owner_id: UUID = Field(...)
    
    # État
    enabled: bool = Field(True)
    last_accessed: Optional[datetime] = Field(None)
    access_count: int = Field(0, ge=0)
    
    def add_widget(self, widget_type: str, config: Dict[str, Any], 
                   position: Optional[Dict[str, int]] = None):
        """Ajoute un widget au tableau de bord"""
        widget = {
            'id': f"widget_{len(self.widgets) + 1}",
            'type': widget_type,
            'config': config,
            'position': position or {'x': 0, 'y': 0, 'width': 4, 'height': 3}
        }
        self.widgets.append(widget)


__all__ = [
    'AnalyticsTimeframe', 'MetricAggregation', 'TrendDirection', 'ReportFormat',
    'PerformanceMetric', 'AlertAnalytics', 'TrendAnalysis', 'AlertReport', 'DashboardConfig'
]
