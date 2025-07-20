"""
Analytics Core Schemas - Ultra-Advanced Edition
=============================================

Schémas ultra-avancés pour l'analytics en temps réel avec validation complexe,
métriques de performance et intégration multi-tenant.

Features:
- Validation de données en temps réel
- Métriques de performance avancées
- Support multi-tenant avec isolation
- Intégration ML/AI native
- Monitoring et alerting intégré
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Set
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, PositiveInt, NonNegativeFloat
import numpy as np
from dataclasses import dataclass


class AnalyticsEventType(str, Enum):
    """Types d'événements analytics supportés."""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    BUSINESS_EVENT = "business_event"
    ML_EVENT = "ml_event"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_EVENT = "performance_event"
    BILLING_EVENT = "billing_event"
    COMPLIANCE_EVENT = "compliance_event"


class AnalyticsChannelType(str, Enum):
    """Canaux de collecte analytics."""
    WEB_APP = "web_app"
    MOBILE_APP = "mobile_app"
    DESKTOP_APP = "desktop_app"
    API = "api"
    WEBHOOK = "webhook"
    STREAMING = "streaming"
    BATCH = "batch"
    REAL_TIME = "real_time"


class DataQualityLevel(str, Enum):
    """Niveaux de qualité des données."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AnalyticsMetadata(BaseModel):
    """Métadonnées avancées pour les événements analytics."""
    
    event_id: UUID4 = Field(..., description="Identifiant unique de l'événement")
    tenant_id: UUID4 = Field(..., description="Identifiant du tenant")
    user_id: Optional[UUID4] = Field(None, description="Identifiant utilisateur")
    session_id: Optional[str] = Field(None, description="Identifiant de session")
    device_id: Optional[str] = Field(None, description="Identifiant du device")
    
    # Métadonnées temporelles
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timezone: str = Field(default="UTC", description="Timezone de l'événement")
    processing_time: Optional[float] = Field(None, description="Temps de traitement en ms")
    
    # Métadonnées techniques
    version: str = Field(default="2.0.0", description="Version du schéma")
    source: AnalyticsChannelType = Field(..., description="Source de l'événement")
    environment: str = Field(default="production", description="Environnement")
    
    # Qualité des données
    data_quality: DataQualityLevel = Field(default=DataQualityLevel.HIGH)
    confidence_score: NonNegativeFloat = Field(default=1.0, ge=0.0, le=1.0)
    
    # Géolocalisation
    country_code: Optional[str] = Field(None, regex=r"^[A-Z]{2}$")
    region: Optional[str] = Field(None)
    city: Optional[str] = Field(None)
    coordinates: Optional[Dict[str, float]] = Field(None)
    
    @validator('coordinates')
    def validate_coordinates(cls, v):
        if v is not None:
            if not isinstance(v, dict) or 'lat' not in v or 'lon' not in v:
                raise ValueError("Coordinates must contain 'lat' and 'lon' keys")
            if not (-90 <= v['lat'] <= 90) or not (-180 <= v['lon'] <= 180):
                raise ValueError("Invalid coordinates range")
        return v


class AnalyticsEvent(BaseModel):
    """Événement analytics ultra-avancé avec validation complète."""
    
    # Métadonnées de base
    metadata: AnalyticsMetadata
    
    # Type et catégorie
    event_type: AnalyticsEventType
    event_name: str = Field(..., min_length=1, max_length=100)
    category: Optional[str] = Field(None, max_length=50)
    subcategory: Optional[str] = Field(None, max_length=50)
    
    # Données de l'événement
    properties: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Union[int, float, Decimal]] = Field(default_factory=dict)
    dimensions: Dict[str, str] = Field(default_factory=dict)
    
    # Contexte utilisateur
    user_agent: Optional[str] = Field(None)
    ip_address: Optional[str] = Field(None)
    referrer: Optional[str] = Field(None)
    
    # Données business
    revenue: Optional[Decimal] = Field(None, description="Revenus générés")
    cost: Optional[Decimal] = Field(None, description="Coût associé")
    conversion_value: Optional[float] = Field(None, description="Valeur de conversion")
    
    # Flags et statuts
    is_test: bool = Field(default=False, description="Événement de test")
    is_synthetic: bool = Field(default=False, description="Données synthétiques")
    is_privacy_compliant: bool = Field(default=True, description="Conformité RGPD")
    
    # Traçabilité
    parent_event_id: Optional[UUID4] = Field(None, description="Événement parent")
    correlation_id: Optional[str] = Field(None, description="ID de corrélation")
    trace_id: Optional[str] = Field(None, description="ID de trace distribuée")
    
    @validator('properties')
    def validate_properties_size(cls, v):
        if len(str(v)) > 10000:  # 10KB limit
            raise ValueError("Properties payload too large")
        return v
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        if v is not None:
            import ipaddress
            try:
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError("Invalid IP address format")
        return v


class AnalyticsQuery(BaseModel):
    """Requête analytics avancée avec optimisations et caching."""
    
    query_id: UUID4 = Field(default_factory=lambda: UUID4())
    tenant_id: UUID4
    user_id: Optional[UUID4] = None
    
    # Paramètres de la requête
    query_type: str = Field(..., description="Type de requête")
    query_string: str = Field(..., min_length=1, description="Requête SQL/NoSQL")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Filtres temporels
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    time_zone: str = Field(default="UTC")
    
    # Filtres de données
    filters: Dict[str, Any] = Field(default_factory=dict)
    dimensions: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    
    # Options de performance
    limit: PositiveInt = Field(default=1000, le=10000)
    offset: int = Field(default=0, ge=0)
    use_cache: bool = Field(default=True)
    cache_ttl: int = Field(default=300, ge=0, le=3600)  # 5 minutes par défaut
    
    # Optimisations
    enable_sampling: bool = Field(default=False)
    sampling_rate: float = Field(default=0.1, ge=0.001, le=1.0)
    enable_approximation: bool = Field(default=False)
    max_execution_time: int = Field(default=30, ge=1, le=300)  # secondes
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if v is not None and 'start_date' in values and values['start_date'] is not None:
            if v <= values['start_date']:
                raise ValueError("End date must be after start date")
            if (v - values['start_date']).days > 365:
                raise ValueError("Date range cannot exceed 365 days")
        return v


class AnalyticsResult(BaseModel):
    """Résultat d'une requête analytics avec métriques de performance."""
    
    query_id: UUID4
    execution_id: UUID4 = Field(default_factory=lambda: UUID4())
    
    # Métadonnées d'exécution
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: float = Field(..., ge=0)
    rows_processed: int = Field(..., ge=0)
    rows_returned: int = Field(..., ge=0)
    
    # Données résultantes
    data: List[Dict[str, Any]] = Field(..., description="Données du résultat")
    columns: List[str] = Field(..., description="Colonnes du résultat")
    total_count: Optional[int] = Field(None, description="Nombre total de lignes")
    
    # Informations de cache
    from_cache: bool = Field(default=False)
    cache_hit_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Métriques de qualité
    data_freshness: Optional[datetime] = Field(None, description="Fraîcheur des données")
    completeness_score: float = Field(default=1.0, ge=0.0, le=1.0)
    accuracy_score: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Warnings et erreurs
    warnings: List[str] = Field(default_factory=list)
    is_approximate: bool = Field(default=False)
    is_sampled: bool = Field(default=False)
    
    # Métadonnées de performance
    memory_usage_mb: Optional[float] = Field(None, ge=0)
    cpu_usage_percent: Optional[float] = Field(None, ge=0, le=100)
    network_io_mb: Optional[float] = Field(None, ge=0)
    
    @validator('data')
    def validate_data_size(cls, v):
        import sys
        size = sys.getsizeof(v)
        if size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError("Result data too large")
        return v


class AnalyticsAggregation(BaseModel):
    """Configuration d'agrégation avancée pour analytics."""
    
    aggregation_id: UUID4 = Field(default_factory=lambda: UUID4())
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    
    # Configuration temporelle
    time_window: str = Field(..., description="Fenêtre temporelle (5m, 1h, 1d, etc.)")
    time_zone: str = Field(default="UTC")
    
    # Métriques à agréger
    metrics: Dict[str, str] = Field(..., description="Métriques et leurs fonctions d'agrégation")
    dimensions: List[str] = Field(..., description="Dimensions pour grouper")
    
    # Filtres
    filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration avancée
    enable_rollup: bool = Field(default=True, description="Activer le rollup")
    retention_days: int = Field(default=90, ge=1, le=3650)
    materialized: bool = Field(default=True, description="Vue matérialisée")
    
    # Alerting
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    alert_operators: Dict[str, str] = Field(default_factory=dict)
    
    @validator('time_window')
    def validate_time_window(cls, v):
        import re
        pattern = r'^(\d+)(s|m|h|d|w)$'
        if not re.match(pattern, v):
            raise ValueError("Invalid time window format. Use format like '5m', '1h', '1d'")
        return v


class AnalyticsReport(BaseModel):
    """Rapport analytics complet avec visualisations."""
    
    report_id: UUID4 = Field(default_factory=lambda: UUID4())
    tenant_id: UUID4
    
    # Métadonnées du rapport
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: UUID4
    
    # Configuration temporelle
    period_start: datetime
    period_end: datetime
    time_zone: str = Field(default="UTC")
    
    # Contenu du rapport
    sections: List[Dict[str, Any]] = Field(..., description="Sections du rapport")
    charts: List[Dict[str, Any]] = Field(default_factory=list, description="Graphiques")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Tableaux")
    
    # Métriques clés
    key_metrics: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
    insights: List[str] = Field(default_factory=list, description="Insights automatiques")
    recommendations: List[str] = Field(default_factory=list, description="Recommandations")
    
    # Configuration de génération
    format: str = Field(default="json", description="Format du rapport")
    auto_refresh: bool = Field(default=False)
    refresh_interval: Optional[int] = Field(None, description="Intervalle de rafraîchissement en minutes")
    
    # Partage et sécurité
    is_public: bool = Field(default=False)
    shared_with: List[UUID4] = Field(default_factory=list)
    access_level: str = Field(default="read", description="Niveau d'accès")
    
    # Métadonnées techniques
    generation_time_ms: Optional[float] = Field(None, ge=0)
    data_freshness: Optional[datetime] = Field(None)
    file_size_bytes: Optional[int] = Field(None, ge=0)
    
    @validator('period_end')
    def validate_period(cls, v, values):
        if 'period_start' in values and v <= values['period_start']:
            raise ValueError("Period end must be after period start")
        return v


# Export des classes principales
__all__ = [
    "AnalyticsEventType",
    "AnalyticsChannelType", 
    "DataQualityLevel",
    "AnalyticsMetadata",
    "AnalyticsEvent",
    "AnalyticsQuery",
    "AnalyticsResult",
    "AnalyticsAggregation",
    "AnalyticsReport"
]
