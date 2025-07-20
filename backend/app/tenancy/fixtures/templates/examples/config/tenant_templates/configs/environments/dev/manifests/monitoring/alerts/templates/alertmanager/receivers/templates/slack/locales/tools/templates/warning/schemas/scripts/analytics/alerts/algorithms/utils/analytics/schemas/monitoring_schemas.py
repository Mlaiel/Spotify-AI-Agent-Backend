"""
Monitoring Schemas - Ultra-Advanced Edition
==========================================

Schémas ultra-avancés pour le monitoring système, applicatif et business
avec alerting intelligent, métriques en temps réel et diagnostics automatisés.

Features:
- Monitoring multi-dimensionnel (système, application, business)
- Alerting intelligent avec ML
- Métriques en temps réel avec agrégation
- Diagnostics automatisés
- SLA et conformité monitoring
- Intégration observabilité complète
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Set, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, PositiveInt, NonNegativeFloat, PositiveFloat
import numpy as np


class MonitoringType(str, Enum):
    """Types de monitoring supportés."""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    SECURITY = "security"
    NETWORK = "network"
    DATABASE = "database"
    USER_EXPERIENCE = "user_experience"
    INFRASTRUCTURE = "infrastructure"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    WARNING = "warning"


class AlertStatus(str, Enum):
    """Statuts des alertes."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"
    CLOSED = "closed"


class MetricType(str, Enum):
    """Types de métriques."""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"
    RATIO = "ratio"
    PERCENTAGE = "percentage"


class ThresholdOperator(str, Enum):
    """Opérateurs pour les seuils."""
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    BETWEEN = "between"
    OUTSIDE = "outside"


class MonitoringTarget(BaseModel):
    """Cible de monitoring avec métadonnées complètes."""
    
    target_id: UUID4 = Field(default_factory=lambda: UUID4())
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    
    # Type et catégorie
    monitoring_type: MonitoringType = Field(..., description="Type de monitoring")
    category: str = Field(..., description="Catégorie de la cible")
    subcategory: Optional[str] = Field(None, description="Sous-catégorie")
    
    # Identification technique
    hostname: Optional[str] = Field(None, description="Nom d'hôte")
    ip_address: Optional[str] = Field(None, description="Adresse IP")
    port: Optional[int] = Field(None, ge=1, le=65535, description="Port")
    service_name: Optional[str] = Field(None, description="Nom du service")
    
    # Métadonnées
    environment: str = Field(default="production", description="Environnement")
    region: Optional[str] = Field(None, description="Région")
    datacenter: Optional[str] = Field(None, description="Centre de données")
    cluster: Optional[str] = Field(None, description="Cluster")
    
    # Configuration
    enabled: bool = Field(default=True, description="Monitoring activé")
    sampling_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Taux d'échantillonnage")
    collection_interval: int = Field(default=60, ge=1, le=3600, description="Intervalle de collecte en secondes")
    
    # Tags et labels
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags personnalisés")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels Prometheus")
    
    # Health check
    health_check_url: Optional[str] = Field(None, description="URL de health check")
    health_check_interval: int = Field(default=30, ge=5, le=300, description="Intervalle health check")
    expected_response_code: int = Field(default=200, description="Code de réponse attendu")
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        if v is not None:
            import ipaddress
            try:
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError("Invalid IP address format")
        return v


class MetricDefinition(BaseModel):
    """Définition d'une métrique avec validation et configuration."""
    
    metric_id: UUID4 = Field(default_factory=lambda: UUID4())
    name: str = Field(..., min_length=1, max_length=100, description="Nom de la métrique")
    display_name: str = Field(..., description="Nom d'affichage")
    description: str = Field(..., max_length=500, description="Description de la métrique")
    
    # Type et unité
    metric_type: MetricType = Field(..., description="Type de métrique")
    unit: str = Field(..., description="Unité de mesure")
    unit_symbol: Optional[str] = Field(None, description="Symbole de l'unité")
    
    # Configuration de collecte
    source_query: str = Field(..., description="Requête source (PromQL, SQL, etc.)")
    collection_method: str = Field(..., description="Méthode de collecte")
    data_source: str = Field(..., description="Source de données")
    
    # Agrégation
    default_aggregation: str = Field(default="avg", description="Agrégation par défaut")
    supported_aggregations: List[str] = Field(default_factory=lambda: ["avg", "sum", "min", "max", "count"])
    
    # Validation des valeurs
    min_value: Optional[float] = Field(None, description="Valeur minimale valide")
    max_value: Optional[float] = Field(None, description="Valeur maximale valide")
    expected_range: Optional[Tuple[float, float]] = Field(None, description="Plage de valeurs attendue")
    
    # Métadonnées
    category: str = Field(..., description="Catégorie de la métrique")
    subcategory: Optional[str] = Field(None, description="Sous-catégorie")
    business_criticality: str = Field(default="medium", description="Criticité business")
    
    # Configuration avancée
    enable_forecasting: bool = Field(default=False, description="Activer la prévision")
    enable_anomaly_detection: bool = Field(default=False, description="Activer la détection d'anomalies")
    enable_trend_analysis: bool = Field(default=False, description="Activer l'analyse de tendance")
    
    # Rétention des données
    retention_days: int = Field(default=90, ge=1, le=3650, description="Rétention en jours")
    high_resolution_days: int = Field(default=7, ge=1, le=90, description="Haute résolution en jours")
    
    @validator('max_value')
    def validate_value_range(cls, v, values):
        if v is not None and 'min_value' in values and values['min_value'] is not None:
            if v <= values['min_value']:
                raise ValueError("Max value must be greater than min value")
        return v


class MonitoringThreshold(BaseModel):
    """Seuil de monitoring avec logique d'alerting avancée."""
    
    threshold_id: UUID4 = Field(default_factory=lambda: UUID4())
    name: str = Field(..., description="Nom du seuil")
    description: Optional[str] = Field(None, description="Description du seuil")
    
    # Configuration du seuil
    metric_name: str = Field(..., description="Nom de la métrique")
    operator: ThresholdOperator = Field(..., description="Opérateur de comparaison")
    value: float = Field(..., description="Valeur seuil")
    secondary_value: Optional[float] = Field(None, description="Valeur secondaire (pour between/outside)")
    
    # Severité et priorité
    severity: AlertSeverity = Field(..., description="Sévérité de l'alerte")
    priority: int = Field(default=5, ge=1, le=10, description="Priorité (1=max, 10=min)")
    
    # Conditions d'activation
    duration_seconds: int = Field(default=300, ge=60, description="Durée avant déclenchement")
    evaluation_period: int = Field(default=60, ge=30, description="Période d'évaluation")
    consecutive_breaches: int = Field(default=1, ge=1, description="Violations consécutives requises")
    
    # Logique avancée
    enable_flap_detection: bool = Field(default=True, description="Détection de flapping")
    flap_threshold_percent: float = Field(default=20.0, ge=0.0, le=100.0)
    enable_seasonality: bool = Field(default=False, description="Prise en compte saisonnalité")
    baseline_days: int = Field(default=7, ge=1, le=30, description="Jours de baseline")
    
    # Conditions contextuelles
    time_ranges: List[Dict[str, str]] = Field(default_factory=list, description="Plages horaires actives")
    day_of_week: Optional[List[int]] = Field(None, description="Jours de la semaine (0=lundi)")
    dependencies: List[str] = Field(default_factory=list, description="Dépendances de seuils")
    
    # Suppression et escalade
    auto_resolve: bool = Field(default=True, description="Résolution automatique")
    auto_resolve_duration: int = Field(default=600, ge=60, description="Durée pour auto-résolution")
    escalation_levels: List[Dict[str, Any]] = Field(default_factory=list, description="Niveaux d'escalade")
    
    @validator('secondary_value')
    def validate_secondary_value(cls, v, values):
        operator = values.get('operator')
        if operator in ['between', 'outside'] and v is None:
            raise ValueError(f"Secondary value required for operator {operator}")
        return v


class SystemMetrics(BaseModel):
    """Métriques système avec monitoring complet."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    target_id: UUID4 = Field(..., description="ID de la cible monitorée")
    
    # CPU
    cpu_usage_percent: NonNegativeFloat = Field(..., le=100.0, description="Usage CPU en %")
    cpu_load_1min: NonNegativeFloat = Field(..., description="Load average 1 minute")
    cpu_load_5min: NonNegativeFloat = Field(..., description="Load average 5 minutes")
    cpu_load_15min: NonNegativeFloat = Field(..., description="Load average 15 minutes")
    cpu_cores: PositiveInt = Field(..., description="Nombre de cœurs CPU")
    
    # Mémoire
    memory_usage_percent: NonNegativeFloat = Field(..., le=100.0, description="Usage mémoire en %")
    memory_used_gb: NonNegativeFloat = Field(..., description="Mémoire utilisée en GB")
    memory_total_gb: PositiveFloat = Field(..., description="Mémoire totale en GB")
    memory_available_gb: NonNegativeFloat = Field(..., description="Mémoire disponible en GB")
    swap_usage_percent: NonNegativeFloat = Field(default=0.0, le=100.0, description="Usage swap en %")
    
    # Stockage
    disk_usage_percent: NonNegativeFloat = Field(..., le=100.0, description="Usage disque en %")
    disk_used_gb: NonNegativeFloat = Field(..., description="Espace disque utilisé en GB")
    disk_total_gb: PositiveFloat = Field(..., description="Espace disque total en GB")
    disk_iops_read: NonNegativeFloat = Field(default=0.0, description="IOPS lecture")
    disk_iops_write: NonNegativeFloat = Field(default=0.0, description="IOPS écriture")
    
    # Réseau
    network_in_mbps: NonNegativeFloat = Field(default=0.0, description="Trafic entrant en Mbps")
    network_out_mbps: NonNegativeFloat = Field(default=0.0, description="Trafic sortant en Mbps")
    network_packets_in: NonNegativeFloat = Field(default=0.0, description="Paquets entrants/sec")
    network_packets_out: NonNegativeFloat = Field(default=0.0, description="Paquets sortants/sec")
    network_errors: NonNegativeFloat = Field(default=0.0, description="Erreurs réseau/sec")
    
    # Processus
    process_count: PositiveInt = Field(..., description="Nombre de processus")
    thread_count: PositiveInt = Field(..., description="Nombre de threads")
    file_descriptors_used: NonNegativeFloat = Field(default=0.0, description="Descripteurs fichiers utilisés")
    file_descriptors_max: PositiveInt = Field(..., description="Descripteurs fichiers max")
    
    # Métadonnées
    uptime_seconds: NonNegativeFloat = Field(..., description="Uptime en secondes")
    boot_time: datetime = Field(..., description="Heure de démarrage")
    kernel_version: Optional[str] = Field(None, description="Version du kernel")
    os_version: Optional[str] = Field(None, description="Version de l'OS")


class ApplicationMetrics(BaseModel):
    """Métriques applicatives avec monitoring APM."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    application_name: str = Field(..., description="Nom de l'application")
    service_name: str = Field(..., description="Nom du service")
    version: str = Field(..., description="Version de l'application")
    
    # Performance
    response_time_ms: NonNegativeFloat = Field(..., description="Temps de réponse moyen en ms")
    response_time_p50: NonNegativeFloat = Field(..., description="Temps de réponse P50")
    response_time_p95: NonNegativeFloat = Field(..., description="Temps de réponse P95")
    response_time_p99: NonNegativeFloat = Field(..., description="Temps de réponse P99")
    
    # Débit
    requests_per_second: NonNegativeFloat = Field(..., description="Requêtes par seconde")
    transactions_per_second: NonNegativeFloat = Field(..., description="Transactions par seconde")
    throughput_mbps: NonNegativeFloat = Field(default=0.0, description="Débit en Mbps")
    
    # Erreurs
    error_rate_percent: NonNegativeFloat = Field(..., le=100.0, description="Taux d'erreur en %")
    http_2xx_count: NonNegativeFloat = Field(default=0.0, description="Réponses 2xx/sec")
    http_3xx_count: NonNegativeFloat = Field(default=0.0, description="Réponses 3xx/sec")
    http_4xx_count: NonNegativeFloat = Field(default=0.0, description="Réponses 4xx/sec")
    http_5xx_count: NonNegativeFloat = Field(default=0.0, description="Réponses 5xx/sec")
    
    # Base de données
    db_connections_active: NonNegativeFloat = Field(default=0.0, description="Connexions DB actives")
    db_connections_idle: NonNegativeFloat = Field(default=0.0, description="Connexions DB inactives")
    db_query_time_ms: NonNegativeFloat = Field(default=0.0, description="Temps requête DB moyen")
    db_slow_queries_count: NonNegativeFloat = Field(default=0.0, description="Requêtes lentes/sec")
    
    # Cache
    cache_hit_ratio: NonNegativeFloat = Field(default=0.0, le=1.0, description="Ratio de cache hit")
    cache_miss_ratio: NonNegativeFloat = Field(default=0.0, le=1.0, description="Ratio de cache miss")
    cache_evictions_count: NonNegativeFloat = Field(default=0.0, description="Évictions cache/sec")
    
    # Garbage Collection (pour langages avec GC)
    gc_collections_count: NonNegativeFloat = Field(default=0.0, description="Collections GC/sec")
    gc_pause_time_ms: NonNegativeFloat = Field(default=0.0, description="Temps pause GC moyen")
    heap_usage_percent: NonNegativeFloat = Field(default=0.0, le=100.0, description="Usage heap en %")
    
    # Business metrics
    active_users: NonNegativeFloat = Field(default=0.0, description="Utilisateurs actifs")
    business_transactions: NonNegativeFloat = Field(default=0.0, description="Transactions business/sec")
    revenue_per_minute: Optional[Decimal] = Field(None, description="Revenus par minute")


class MonitoringAlert(BaseModel):
    """Alerte de monitoring avec workflow complet."""
    
    # Identification
    alert_id: UUID4 = Field(default_factory=lambda: UUID4())
    correlation_id: Optional[str] = Field(None, description="ID de corrélation")
    
    # Métadonnées de base
    title: str = Field(..., min_length=1, max_length=200, description="Titre de l'alerte")
    description: str = Field(..., max_length=1000, description="Description détaillée")
    
    # Classification
    severity: AlertSeverity = Field(..., description="Sévérité de l'alerte")
    category: str = Field(..., description="Catégorie de l'alerte")
    subcategory: Optional[str] = Field(None, description="Sous-catégorie")
    alert_type: str = Field(..., description="Type d'alerte")
    
    # Source et cible
    source_system: str = Field(..., description="Système source")
    target_id: UUID4 = Field(..., description="ID de la cible")
    target_name: str = Field(..., description="Nom de la cible")
    
    # Timing
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    first_occurrence: datetime = Field(default_factory=datetime.utcnow)
    last_occurrence: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = Field(None, description="Heure de résolution")
    
    # Statut et workflow
    status: AlertStatus = Field(default=AlertStatus.ACTIVE)
    acknowledged_by: Optional[UUID4] = Field(None, description="Utilisateur qui a acquitté")
    acknowledged_at: Optional[datetime] = Field(None, description="Heure d'acquittement")
    assigned_to: Optional[UUID4] = Field(None, description="Utilisateur assigné")
    
    # Détails techniques
    metric_name: str = Field(..., description="Nom de la métrique")
    threshold_value: float = Field(..., description="Valeur seuil")
    current_value: float = Field(..., description="Valeur actuelle")
    operator: str = Field(..., description="Opérateur de comparaison")
    
    # Contexte
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags de l'alerte")
    environment: str = Field(..., description="Environnement")
    region: Optional[str] = Field(None, description="Région")
    
    # Escalade
    escalation_level: int = Field(default=0, ge=0, description="Niveau d'escalade")
    escalated_at: Optional[datetime] = Field(None, description="Heure d'escalade")
    escalation_reason: Optional[str] = Field(None, description="Raison d'escalade")
    
    # Suppression
    suppressed: bool = Field(default=False, description="Alerte supprimée")
    suppressed_until: Optional[datetime] = Field(None, description="Supprimée jusqu'à")
    suppression_reason: Optional[str] = Field(None, description="Raison de suppression")
    
    # Notification
    notification_sent: bool = Field(default=False, description="Notification envoyée")
    notification_channels: List[str] = Field(default_factory=list, description="Canaux de notification")
    notification_count: int = Field(default=0, ge=0, description="Nombre de notifications")
    
    # Diagnostic automatique
    probable_cause: Optional[str] = Field(None, description="Cause probable (IA)")
    suggested_actions: List[str] = Field(default_factory=list, description="Actions suggérées")
    similar_alerts: List[UUID4] = Field(default_factory=list, description="Alertes similaires")
    
    # Impact business
    business_impact: Optional[str] = Field(None, description="Impact business")
    affected_services: List[str] = Field(default_factory=list, description="Services affectés")
    user_impact_count: Optional[int] = Field(None, description="Nombre d'utilisateurs impactés")
    
    # Métadonnées de résolution
    resolution_method: Optional[str] = Field(None, description="Méthode de résolution")
    resolution_notes: Optional[str] = Field(None, description="Notes de résolution")
    auto_resolved: bool = Field(default=False, description="Résolution automatique")
    
    @validator('resolved_at')
    def validate_resolution_time(cls, v, values):
        if v is not None and 'triggered_at' in values:
            if v < values['triggered_at']:
                raise ValueError("Resolution time cannot be before trigger time")
        return v
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Calcule la durée de l'alerte en minutes."""
        end_time = self.resolved_at or datetime.utcnow()
        return (end_time - self.triggered_at).total_seconds() / 60


class HealthCheck(BaseModel):
    """Health check complet avec diagnostics avancés."""
    
    check_id: UUID4 = Field(default_factory=lambda: UUID4())
    target_id: UUID4 = Field(..., description="ID de la cible")
    
    # Métadonnées
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    check_type: str = Field(..., description="Type de health check")
    check_name: str = Field(..., description="Nom du check")
    
    # Résultat global
    healthy: bool = Field(..., description="État de santé global")
    status: str = Field(..., description="Statut détaillé")
    
    # Métriques de performance
    response_time_ms: NonNegativeFloat = Field(..., description="Temps de réponse")
    timeout_ms: PositiveInt = Field(default=5000, description="Timeout configuré")
    
    # Checks détaillés
    component_checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Checks par composant")
    dependency_checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Checks dépendances")
    
    # Diagnostics
    error_message: Optional[str] = Field(None, description="Message d'erreur")
    warnings: List[str] = Field(default_factory=list, description="Avertissements")
    diagnostic_info: Dict[str, Any] = Field(default_factory=dict, description="Infos diagnostiques")
    
    # Tendances
    success_rate_1h: Optional[float] = Field(None, ge=0.0, le=1.0, description="Taux de succès 1h")
    success_rate_24h: Optional[float] = Field(None, ge=0.0, le=1.0, description="Taux de succès 24h")
    avg_response_time_1h: Optional[float] = Field(None, description="Temps réponse moyen 1h")
    
    # Métadonnées techniques
    check_version: str = Field(default="1.0.0", description="Version du check")
    environment: str = Field(..., description="Environnement")
    source_ip: Optional[str] = Field(None, description="IP source du check")


# Export des classes principales
__all__ = [
    "MonitoringType",
    "AlertSeverity",
    "AlertStatus", 
    "MetricType",
    "ThresholdOperator",
    "MonitoringTarget",
    "MetricDefinition",
    "MonitoringThreshold",
    "SystemMetrics",
    "ApplicationMetrics",
    "MonitoringAlert",
    "HealthCheck"
]
