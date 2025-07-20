"""
Schémas de métriques pour les alertes - Spotify AI Agent
Collecte et analyse des métriques système et business
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from decimal import Decimal
from statistics import mean, median, stdev

from pydantic import BaseModel, Field, validator, computed_field
from pydantic.types import StrictStr

from ..base import BaseSchema, TimestampMixin
from ..base.enums import (
    AlertLevel, AlertStatus, WarningCategory, Environment,
    MetricType, TimeUnit, DataSize
)


class MetricValue(BaseModel):
    """Valeur de métrique avec timestamp"""
    timestamp: datetime = Field(...)
    value: Union[float, int, str, bool] = Field(...)
    unit: Optional[str] = Field(None)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricSeries(BaseSchema):
    """Série temporelle de métriques"""
    metric_name: StrictStr = Field(..., min_length=1, max_length=255)
    metric_type: MetricType = Field(...)
    description: Optional[str] = Field(None, max_length=500)
    
    # Configuration
    unit: Optional[str] = Field(None, max_length=50)
    precision: int = Field(2, ge=0, le=10)
    
    # Données
    values: List[MetricValue] = Field(default_factory=list)
    total_points: int = Field(0, ge=0)
    
    # Métadonnées temporelles
    start_time: Optional[datetime] = Field(None)
    end_time: Optional[datetime] = Field(None)
    interval_seconds: Optional[int] = Field(None, ge=1)
    
    # Labels et dimensions
    static_labels: Dict[str, str] = Field(default_factory=dict)
    dimensions: List[str] = Field(default_factory=list)
    
    @validator('values')
    def validate_values(cls, v):
        """Valide les valeurs de métriques"""
        if len(v) > 10000:  # Limite pour éviter la surcharge
            raise ValueError('Too many metric values (max 10000)')
        return v
    
    @computed_field
    @property
    def latest_value(self) -> Optional[MetricValue]:
        """Dernière valeur de la série"""
        return self.values[-1] if self.values else None
    
    @computed_field
    @property
    def duration_seconds(self) -> Optional[int]:
        """Durée de la série en secondes"""
        if not self.start_time or not self.end_time:
            return None
        return int((self.end_time - self.start_time).total_seconds())
    
    def add_value(self, value: Union[float, int, str, bool], 
                  timestamp: Optional[datetime] = None,
                  labels: Optional[Dict[str, str]] = None):
        """Ajoute une valeur à la série"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        metric_value = MetricValue(
            timestamp=timestamp,
            value=value,
            unit=self.unit,
            labels=labels or {}
        )
        
        self.values.append(metric_value)
        self.total_points += 1
        
        # Mise à jour des bornes temporelles
        if not self.start_time or timestamp < self.start_time:
            self.start_time = timestamp
        if not self.end_time or timestamp > self.end_time:
            self.end_time = timestamp
    
    def get_numeric_values(self) -> List[float]:
        """Extrait les valeurs numériques"""
        numeric_values = []
        for value in self.values:
            if isinstance(value.value, (int, float)):
                numeric_values.append(float(value.value))
        return numeric_values
    
    def calculate_statistics(self) -> Optional[Dict[str, float]]:
        """Calcule les statistiques de base"""
        numeric_values = self.get_numeric_values()
        if not numeric_values:
            return None
        
        stats = {
            'count': len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': mean(numeric_values),
            'median': median(numeric_values)
        }
        
        if len(numeric_values) > 1:
            stats['stddev'] = stdev(numeric_values)
        
        return stats


class AlertMetrics(BaseSchema):
    """Métriques agrégées d'alertes"""
    # Période de calcul
    period_start: datetime = Field(...)
    period_end: datetime = Field(...)
    calculation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Compteurs globaux
    total_alerts: int = Field(0, ge=0)
    new_alerts: int = Field(0, ge=0)
    resolved_alerts: int = Field(0, ge=0)
    acknowledged_alerts: int = Field(0, ge=0)
    escalated_alerts: int = Field(0, ge=0)
    suppressed_alerts: int = Field(0, ge=0)
    
    # Répartition par niveau
    alerts_by_level: Dict[AlertLevel, int] = Field(default_factory=dict)
    
    # Répartition par catégorie
    alerts_by_category: Dict[WarningCategory, int] = Field(default_factory=dict)
    
    # Répartition par statut
    alerts_by_status: Dict[AlertStatus, int] = Field(default_factory=dict)
    
    # Répartition par environnement
    alerts_by_environment: Dict[Environment, int] = Field(default_factory=dict)
    
    # Métriques temporelles (en secondes)
    avg_time_to_acknowledge: Optional[float] = Field(None, ge=0)
    avg_time_to_resolve: Optional[float] = Field(None, ge=0)
    avg_alert_duration: Optional[float] = Field(None, ge=0)
    median_time_to_acknowledge: Optional[float] = Field(None, ge=0)
    median_time_to_resolve: Optional[float] = Field(None, ge=0)
    
    # Métriques de qualité
    false_positive_rate: Optional[float] = Field(None, ge=0, le=1)
    escalation_rate: Optional[float] = Field(None, ge=0, le=1)
    resolution_rate: Optional[float] = Field(None, ge=0, le=1)
    suppression_rate: Optional[float] = Field(None, ge=0, le=1)
    
    # Top services/tenants
    top_services_by_alerts: List[Dict[str, Union[str, int]]] = Field(default_factory=list)
    top_tenants_by_alerts: List[Dict[str, Union[str, int]]] = Field(default_factory=list)
    
    # Tendances (comparaison avec période précédente)
    alert_count_trend: Optional[float] = Field(None, description="% de changement")
    resolution_time_trend: Optional[float] = Field(None, description="% de changement")
    
    @validator('period_end')
    def validate_period(cls, v, values):
        """Valide la cohérence de la période"""
        start = values.get('period_start')
        if start and v <= start:
            raise ValueError('period_end must be after period_start')
        return v
    
    @computed_field
    @property
    def period_duration_hours(self) -> float:
        """Durée de la période en heures"""
        return (self.period_end - self.period_start).total_seconds() / 3600
    
    @computed_field
    @property
    def alerts_per_hour(self) -> float:
        """Taux d'alertes par heure"""
        duration = self.period_duration_hours
        return self.total_alerts / duration if duration > 0 else 0
    
    @computed_field
    @property
    def critical_alert_percentage(self) -> float:
        """Pourcentage d'alertes critiques"""
        if self.total_alerts == 0:
            return 0
        critical_count = self.alerts_by_level.get(AlertLevel.CRITICAL, 0)
        return (critical_count / self.total_alerts) * 100


class SystemMetrics(BaseSchema):
    """Métriques système pour monitoring"""
    # Métriques CPU
    cpu_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    cpu_load_average_1m: Optional[float] = Field(None, ge=0)
    cpu_load_average_5m: Optional[float] = Field(None, ge=0)
    cpu_load_average_15m: Optional[float] = Field(None, ge=0)
    cpu_cores: Optional[int] = Field(None, ge=1)
    
    # Métriques mémoire
    memory_total_bytes: Optional[int] = Field(None, ge=0)
    memory_used_bytes: Optional[int] = Field(None, ge=0)
    memory_available_bytes: Optional[int] = Field(None, ge=0)
    memory_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    memory_swap_used_bytes: Optional[int] = Field(None, ge=0)
    
    # Métriques disque
    disk_total_bytes: Optional[int] = Field(None, ge=0)
    disk_used_bytes: Optional[int] = Field(None, ge=0)
    disk_available_bytes: Optional[int] = Field(None, ge=0)
    disk_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    disk_io_read_bytes_per_sec: Optional[float] = Field(None, ge=0)
    disk_io_write_bytes_per_sec: Optional[float] = Field(None, ge=0)
    
    # Métriques réseau
    network_rx_bytes_per_sec: Optional[float] = Field(None, ge=0)
    network_tx_bytes_per_sec: Optional[float] = Field(None, ge=0)
    network_rx_packets_per_sec: Optional[float] = Field(None, ge=0)
    network_tx_packets_per_sec: Optional[float] = Field(None, ge=0)
    network_connections_active: Optional[int] = Field(None, ge=0)
    
    # Métriques processus
    process_count: Optional[int] = Field(None, ge=0)
    thread_count: Optional[int] = Field(None, ge=0)
    file_descriptors_used: Optional[int] = Field(None, ge=0)
    file_descriptors_max: Optional[int] = Field(None, ge=0)
    
    # Uptime
    uptime_seconds: Optional[float] = Field(None, ge=0)
    
    @computed_field
    @property
    def memory_usage_ratio(self) -> Optional[float]:
        """Ratio d'utilisation mémoire"""
        if self.memory_total_bytes and self.memory_used_bytes:
            return self.memory_used_bytes / self.memory_total_bytes
        return None
    
    @computed_field
    @property
    def disk_usage_ratio(self) -> Optional[float]:
        """Ratio d'utilisation disque"""
        if self.disk_total_bytes and self.disk_used_bytes:
            return self.disk_used_bytes / self.disk_total_bytes
        return None
    
    @computed_field
    @property
    def system_health_score(self) -> float:
        """Score de santé système (0-100)"""
        score = 100.0
        
        # Pénalité CPU
        if self.cpu_usage_percent:
            if self.cpu_usage_percent > 90:
                score -= 30
            elif self.cpu_usage_percent > 80:
                score -= 20
            elif self.cpu_usage_percent > 70:
                score -= 10
        
        # Pénalité mémoire
        if self.memory_usage_percent:
            if self.memory_usage_percent > 95:
                score -= 25
            elif self.memory_usage_percent > 85:
                score -= 15
            elif self.memory_usage_percent > 75:
                score -= 5
        
        # Pénalité disque
        if self.disk_usage_percent:
            if self.disk_usage_percent > 95:
                score -= 20
            elif self.disk_usage_percent > 90:
                score -= 10
            elif self.disk_usage_percent > 85:
                score -= 5
        
        return max(score, 0.0)


class ApplicationMetrics(BaseSchema):
    """Métriques applicatives"""
    # Métriques HTTP
    http_requests_total: Optional[int] = Field(None, ge=0)
    http_requests_per_second: Optional[float] = Field(None, ge=0)
    http_response_time_ms_avg: Optional[float] = Field(None, ge=0)
    http_response_time_ms_p50: Optional[float] = Field(None, ge=0)
    http_response_time_ms_p95: Optional[float] = Field(None, ge=0)
    http_response_time_ms_p99: Optional[float] = Field(None, ge=0)
    
    # Codes de statut HTTP
    http_status_2xx_count: Optional[int] = Field(None, ge=0)
    http_status_3xx_count: Optional[int] = Field(None, ge=0)
    http_status_4xx_count: Optional[int] = Field(None, ge=0)
    http_status_5xx_count: Optional[int] = Field(None, ge=0)
    
    # Métriques base de données
    db_connections_active: Optional[int] = Field(None, ge=0)
    db_connections_max: Optional[int] = Field(None, ge=0)
    db_query_time_ms_avg: Optional[float] = Field(None, ge=0)
    db_query_time_ms_p95: Optional[float] = Field(None, ge=0)
    db_transactions_per_second: Optional[float] = Field(None, ge=0)
    db_deadlocks_count: Optional[int] = Field(None, ge=0)
    
    # Métriques cache
    cache_hit_rate: Optional[float] = Field(None, ge=0, le=1)
    cache_miss_rate: Optional[float] = Field(None, ge=0, le=1)
    cache_evictions_per_second: Optional[float] = Field(None, ge=0)
    cache_memory_usage_bytes: Optional[int] = Field(None, ge=0)
    
    # Métriques queue/messaging
    queue_depth: Optional[int] = Field(None, ge=0)
    queue_processing_rate: Optional[float] = Field(None, ge=0)
    message_processing_time_ms: Optional[float] = Field(None, ge=0)
    failed_messages_count: Optional[int] = Field(None, ge=0)
    
    # Métriques business
    active_users: Optional[int] = Field(None, ge=0)
    active_sessions: Optional[int] = Field(None, ge=0)
    business_transactions_per_minute: Optional[float] = Field(None, ge=0)
    revenue_per_hour: Optional[Decimal] = Field(None, ge=0)
    
    @computed_field
    @property
    def error_rate(self) -> Optional[float]:
        """Taux d'erreur HTTP"""
        total_errors = (self.http_status_4xx_count or 0) + (self.http_status_5xx_count or 0)
        total_requests = self.http_requests_total
        
        if total_requests and total_requests > 0:
            return total_errors / total_requests
        return None
    
    @computed_field
    @property
    def db_connection_usage_ratio(self) -> Optional[float]:
        """Ratio d'utilisation des connexions DB"""
        if self.db_connections_max and self.db_connections_active:
            return self.db_connections_active / self.db_connections_max
        return None


class MetricThreshold(BaseModel):
    """Seuil de métrique pour alerting"""
    metric_name: StrictStr = Field(...)
    threshold_value: float = Field(...)
    operator: str = Field(..., regex=r"^(>|>=|<|<=|==|!=)$")
    duration_seconds: int = Field(60, ge=1, le=86400)  # 1 sec à 24h
    severity: AlertLevel = Field(AlertLevel.WARNING)
    enabled: bool = Field(True)
    
    # Configuration avancée
    hysteresis_percent: Optional[float] = Field(None, ge=0, le=100)
    evaluation_interval_seconds: int = Field(30, ge=1, le=3600)
    consecutive_breaches_required: int = Field(1, ge=1, le=100)
    
    # Contexte
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    def evaluate(self, value: float) -> bool:
        """Évalue si le seuil est dépassé"""
        if self.operator == ">":
            return value > self.threshold_value
        elif self.operator == ">=":
            return value >= self.threshold_value
        elif self.operator == "<":
            return value < self.threshold_value
        elif self.operator == "<=":
            return value <= self.threshold_value
        elif self.operator == "==":
            return value == self.threshold_value
        elif self.operator == "!=":
            return value != self.threshold_value
        return False


class MetricAggregation(BaseModel):
    """Agrégation de métriques"""
    metric_name: str = Field(...)
    aggregation_type: str = Field(..., regex=r"^(sum|avg|min|max|count|rate)$")
    time_window_seconds: int = Field(300, ge=1, le=86400)  # 5 min par défaut
    
    # Filtres
    label_filters: Dict[str, str] = Field(default_factory=dict)
    
    # Résultats
    result_value: Optional[float] = Field(None)
    data_points_count: int = Field(0, ge=0)
    calculation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
