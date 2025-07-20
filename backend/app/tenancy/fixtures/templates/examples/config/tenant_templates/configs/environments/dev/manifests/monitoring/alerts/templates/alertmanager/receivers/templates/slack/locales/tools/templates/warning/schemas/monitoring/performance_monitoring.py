"""
Advanced Performance Monitoring - Industrial Grade APM & Observability System
============================================================================

Ce module fournit une architecture de monitoring de performance ultra-avancée
pour observabilité complète des applications, infrastructure et expérience utilisateur.

Features:
- Application Performance Monitoring (APM)
- Real User Monitoring (RUM)
- Synthetic monitoring and testing
- Infrastructure performance tracking
- Database and query optimization
- Distributed tracing and profiling
- Business transaction monitoring
- SLA/SLO monitoring and alerting
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import json
from uuid import uuid4


class ServiceTier(str, Enum):
    """Niveaux de service"""
    CRITICAL = "critical"           # Service critique business
    IMPORTANT = "important"         # Service important
    STANDARD = "standard"          # Service standard
    DEVELOPMENT = "development"     # Service développement


class PerformanceStatus(str, Enum):
    """Statuts de performance"""
    EXCELLENT = "excellent"        # Performance excellente
    GOOD = "good"                  # Performance bonne
    ACCEPTABLE = "acceptable"      # Performance acceptable
    DEGRADED = "degraded"          # Performance dégradée
    CRITICAL = "critical"          # Performance critique
    UNKNOWN = "unknown"            # Statut inconnu


class TransactionType(str, Enum):
    """Types de transactions business"""
    USER_REGISTRATION = "user_registration"
    LOGIN = "login"
    SEARCH = "search"
    RECOMMENDATION = "recommendation"
    PLAYBACK = "playback"
    PAYMENT = "payment"
    PLAYLIST_CREATION = "playlist_creation"
    SOCIAL_INTERACTION = "social_interaction"
    API_REQUEST = "api_request"
    DATA_PROCESSING = "data_processing"


class AlertThreshold(BaseModel):
    """Seuil d'alerte pour métriques de performance"""
    metric_name: str = Field(..., description="Nom de la métrique")
    warning_threshold: float = Field(..., description="Seuil d'avertissement")
    critical_threshold: float = Field(..., description="Seuil critique")
    unit: str = Field(..., description="Unité de mesure")
    direction: str = Field("above", description="Direction (above/below)")
    
    @validator('direction')
    def validate_direction(cls, v):
        if v not in ['above', 'below']:
            raise ValueError('Direction must be "above" or "below"')
        return v


class SLO(BaseModel):
    """Service Level Objective"""
    
    # Identifiants
    slo_id: str = Field(..., description="ID unique SLO")
    name: str = Field(..., description="Nom SLO")
    description: str = Field(..., description="Description")
    
    # Service associé
    service_name: str = Field(..., description="Nom du service")
    service_tier: ServiceTier = Field(..., description="Niveau du service")
    
    # Objectif
    target_percentage: float = Field(..., description="Pourcentage cible (ex: 99.9)")
    measurement_window: str = Field(..., description="Fenêtre de mesure (ex: 30d)")
    
    # SLI (Service Level Indicator)
    sli_metric: str = Field(..., description="Métrique SLI")
    sli_query: str = Field(..., description="Requête SLI")
    good_event_query: str = Field(..., description="Requête événements bons")
    total_event_query: str = Field(..., description="Requête événements totaux")
    
    # Budgets d'erreur
    error_budget_consumed: float = Field(0.0, description="Budget erreur consommé (%)")
    error_budget_remaining: float = Field(100.0, description="Budget erreur restant (%)")
    
    # Alertes
    alerting_enabled: bool = Field(True, description="Alertes activées")
    burn_rate_alerts: List[AlertThreshold] = Field(
        default_factory=list, description="Alertes burn rate"
    )
    
    # Responsabilités
    owner: str = Field(..., description="Propriétaire SLO")
    team: str = Field(..., description="Équipe responsable")
    
    # Métadonnées
    tags: List[str] = Field(default_factory=list, description="Tags")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('target_percentage', 'error_budget_consumed', 'error_budget_remaining')
    def validate_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Percentage must be between 0 and 100')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "slo_id": "spotify_api_availability",
                "name": "Spotify API Availability",
                "service_name": "spotify-api",
                "service_tier": "critical",
                "target_percentage": 99.9,
                "measurement_window": "30d",
                "sli_metric": "availability"
            }
        }


class ApplicationMetrics(BaseModel):
    """Métriques de performance d'application"""
    
    # Identifiants
    service_name: str = Field(..., description="Nom du service")
    instance_id: str = Field(..., description="ID instance")
    
    # Métriques de latence
    avg_response_time_ms: float = Field(..., description="Temps réponse moyen (ms)")
    p50_response_time_ms: float = Field(..., description="Temps réponse P50 (ms)")
    p95_response_time_ms: float = Field(..., description="Temps réponse P95 (ms)")
    p99_response_time_ms: float = Field(..., description="Temps réponse P99 (ms)")
    
    # Métriques de débit
    requests_per_second: float = Field(..., description="Requêtes par seconde")
    successful_requests_per_second: float = Field(..., description="Requêtes réussies/sec")
    failed_requests_per_second: float = Field(..., description="Requêtes échouées/sec")
    
    # Métriques d'erreur
    error_rate_percentage: float = Field(..., description="Taux d'erreur (%)")
    http_4xx_rate: float = Field(0.0, description="Taux erreurs 4xx")
    http_5xx_rate: float = Field(0.0, description="Taux erreurs 5xx")
    timeout_rate: float = Field(0.0, description="Taux timeouts")
    
    # Métriques de ressources
    cpu_usage_percentage: float = Field(..., description="Utilisation CPU (%)")
    memory_usage_percentage: float = Field(..., description="Utilisation mémoire (%)")
    memory_usage_mb: float = Field(..., description="Utilisation mémoire (MB)")
    
    # Métriques de connectivité
    active_connections: int = Field(..., description="Connexions actives")
    connection_pool_usage: float = Field(0.0, description="Utilisation pool connexions (%)")
    
    # Métriques de cache
    cache_hit_rate: float = Field(0.0, description="Taux cache hit (%)")
    cache_miss_rate: float = Field(0.0, description="Taux cache miss (%)")
    
    # Métriques business
    business_transactions_per_second: float = Field(0.0, description="Transactions business/sec")
    revenue_impacting_errors: int = Field(0, description="Erreurs impactant revenus")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('error_rate_percentage', 'cpu_usage_percentage', 'memory_usage_percentage', 
              'cache_hit_rate', 'cache_miss_rate', 'connection_pool_usage')
    def validate_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Percentage must be between 0 and 100')
        return v


class DatabaseMetrics(BaseModel):
    """Métriques de performance base de données"""
    
    # Identifiants
    database_name: str = Field(..., description="Nom base de données")
    database_type: str = Field(..., description="Type de BDD (PostgreSQL, Redis, etc.)")
    instance_id: str = Field(..., description="ID instance")
    
    # Métriques de performance
    avg_query_time_ms: float = Field(..., description="Temps requête moyen (ms)")
    slow_queries_count: int = Field(0, description="Nombre requêtes lentes")
    blocked_queries_count: int = Field(0, description="Requêtes bloquées")
    
    # Métriques de connectivité
    active_connections: int = Field(..., description="Connexions actives")
    max_connections: int = Field(..., description="Connexions maximum")
    connection_usage_percentage: float = Field(..., description="Utilisation connexions (%)")
    
    # Métriques de stockage
    database_size_gb: float = Field(..., description="Taille BDD (GB)")
    storage_usage_percentage: float = Field(..., description="Utilisation stockage (%)")
    index_size_gb: float = Field(0.0, description="Taille index (GB)")
    
    # Métriques de cache
    buffer_cache_hit_ratio: float = Field(0.0, description="Taux cache buffer (%)")
    query_cache_hit_ratio: float = Field(0.0, description="Taux cache requêtes (%)")
    
    # Métriques de réplication
    replication_lag_ms: Optional[float] = Field(None, description="Lag réplication (ms)")
    replica_count: int = Field(0, description="Nombre répliques")
    
    # Métriques de verrouillage
    lock_waits_count: int = Field(0, description="Attentes verrous")
    deadlock_count: int = Field(0, description="Deadlocks")
    
    # Métriques spécifiques Redis
    memory_usage_mb: Optional[float] = Field(None, description="Utilisation mémoire Redis (MB)")
    keyspace_hit_ratio: Optional[float] = Field(None, description="Taux hit keyspace (%)")
    expired_keys_count: Optional[int] = Field(None, description="Clés expirées")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('connection_usage_percentage', 'storage_usage_percentage', 
              'buffer_cache_hit_ratio', 'query_cache_hit_ratio', 'keyspace_hit_ratio')
    def validate_percentage(cls, v):
        if v is not None and not 0 <= v <= 100:
            raise ValueError('Percentage must be between 0 and 100')
        return v


class InfrastructureMetrics(BaseModel):
    """Métriques d'infrastructure"""
    
    # Identifiants
    node_name: str = Field(..., description="Nom du noeud")
    node_type: str = Field(..., description="Type de noeud (worker, master, etc.)")
    zone: str = Field(..., description="Zone de disponibilité")
    
    # Métriques CPU
    cpu_usage_percentage: float = Field(..., description="Utilisation CPU (%)")
    cpu_load_1min: float = Field(..., description="Load average 1min")
    cpu_load_5min: float = Field(..., description="Load average 5min")
    cpu_load_15min: float = Field(..., description="Load average 15min")
    
    # Métriques mémoire
    memory_usage_percentage: float = Field(..., description="Utilisation mémoire (%)")
    memory_used_gb: float = Field(..., description="Mémoire utilisée (GB)")
    memory_available_gb: float = Field(..., description="Mémoire disponible (GB)")
    swap_usage_percentage: float = Field(0.0, description="Utilisation swap (%)")
    
    # Métriques disque
    disk_usage_percentage: float = Field(..., description="Utilisation disque (%)")
    disk_io_read_mb_per_sec: float = Field(0.0, description="IO lecture (MB/s)")
    disk_io_write_mb_per_sec: float = Field(0.0, description="IO écriture (MB/s)")
    disk_iops_read: float = Field(0.0, description="IOPS lecture")
    disk_iops_write: float = Field(0.0, description="IOPS écriture")
    
    # Métriques réseau
    network_in_mb_per_sec: float = Field(0.0, description="Trafic entrant (MB/s)")
    network_out_mb_per_sec: float = Field(0.0, description="Trafic sortant (MB/s)")
    network_packets_in_per_sec: float = Field(0.0, description="Paquets entrants/sec")
    network_packets_out_per_sec: float = Field(0.0, description="Paquets sortants/sec")
    
    # Métriques Kubernetes
    pod_count: Optional[int] = Field(None, description="Nombre de pods")
    container_count: Optional[int] = Field(None, description="Nombre conteneurs")
    pod_restart_count: Optional[int] = Field(None, description="Redémarrages pods")
    
    # État du système
    uptime_hours: float = Field(..., description="Uptime (heures)")
    temperature_celsius: Optional[float] = Field(None, description="Température (°C)")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('cpu_usage_percentage', 'memory_usage_percentage', 'swap_usage_percentage', 'disk_usage_percentage')
    def validate_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Percentage must be between 0 and 100')
        return v


class BusinessTransaction(BaseModel):
    """Transaction business pour monitoring"""
    
    # Identifiants
    transaction_id: str = Field(default_factory=lambda: str(uuid4()), description="ID transaction")
    transaction_type: TransactionType = Field(..., description="Type transaction")
    user_id: Optional[str] = Field(None, description="ID utilisateur")
    
    # Timing
    start_time: datetime = Field(..., description="Début transaction")
    end_time: Optional[datetime] = Field(None, description="Fin transaction")
    duration_ms: Optional[float] = Field(None, description="Durée (ms)")
    
    # Statut
    status: str = Field(..., description="Statut (success/failure/timeout)")
    error_code: Optional[str] = Field(None, description="Code erreur")
    error_message: Optional[str] = Field(None, description="Message erreur")
    
    # Métriques
    steps_completed: int = Field(0, description="Étapes complétées")
    total_steps: int = Field(1, description="Total étapes")
    
    # Business context
    revenue_impact: Optional[float] = Field(None, description="Impact revenus")
    user_satisfaction_score: Optional[float] = Field(None, description="Score satisfaction")
    
    # Technique
    service_calls: List[str] = Field(default_factory=list, description="Appels services")
    database_queries: int = Field(0, description="Requêtes BDD")
    cache_hits: int = Field(0, description="Cache hits")
    cache_misses: int = Field(0, description="Cache misses")
    
    # Géolocalisation
    country: Optional[str] = Field(None, description="Pays")
    region: Optional[str] = Field(None, description="Région")
    
    # Device/Platform
    platform: Optional[str] = Field(None, description="Plateforme")
    device_type: Optional[str] = Field(None, description="Type appareil")
    
    # Tags
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags")


class SyntheticTest(BaseModel):
    """Test synthétique de monitoring"""
    
    # Identifiants
    test_id: str = Field(..., description="ID test")
    test_name: str = Field(..., description="Nom test")
    test_type: str = Field(..., description="Type test (http, browser, api)")
    
    # Configuration
    target_url: str = Field(..., description="URL cible")
    test_frequency: str = Field("5m", description="Fréquence test")
    timeout_ms: int = Field(30000, description="Timeout (ms)")
    
    # Locations
    test_locations: List[str] = Field(..., description="Localisations test")
    
    # Résultats
    last_run_time: datetime = Field(..., description="Dernière exécution")
    success_rate_24h: float = Field(..., description="Taux succès 24h (%)")
    avg_response_time_ms: float = Field(..., description="Temps réponse moyen (ms)")
    
    # Détails dernière exécution
    last_result_status: str = Field(..., description="Statut dernier résultat")
    last_result_time_ms: Optional[float] = Field(None, description="Temps dernier résultat (ms)")
    last_error_message: Optional[str] = Field(None, description="Dernier message erreur")
    
    # Alertes
    alerting_enabled: bool = Field(True, description="Alertes activées")
    failure_threshold: int = Field(3, description="Seuil échecs consécutifs")
    
    # Tags
    tags: List[str] = Field(default_factory=list, description="Tags")
    
    @validator('success_rate_24h')
    def validate_success_rate(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Success rate must be between 0 and 100')
        return v


class PerformanceProfile(BaseModel):
    """Profil de performance d'un service"""
    
    # Identifiants
    service_name: str = Field(..., description="Nom service")
    profile_period: str = Field(..., description="Période profil")
    
    # Performance globale
    overall_status: PerformanceStatus = Field(..., description="Statut global")
    performance_score: float = Field(..., description="Score performance (0-100)")
    
    # Métriques agrégées
    avg_response_time_ms: float = Field(..., description="Temps réponse moyen (ms)")
    error_rate_percentage: float = Field(..., description="Taux erreur (%)")
    throughput_rps: float = Field(..., description="Débit (req/s)")
    availability_percentage: float = Field(..., description="Disponibilité (%)")
    
    # Tendances
    response_time_trend: str = Field(..., description="Tendance temps réponse")
    error_rate_trend: str = Field(..., description="Tendance taux erreur")
    throughput_trend: str = Field(..., description="Tendance débit")
    
    # Problèmes identifiés
    performance_issues: List[str] = Field(default_factory=list, description="Problèmes performance")
    recommendations: List[str] = Field(default_factory=list, description="Recommandations")
    
    # SLO compliance
    slo_compliance_percentage: float = Field(0.0, description="Conformité SLO (%)")
    slo_violations_count: int = Field(0, description="Violations SLO")
    
    # Resource utilization
    avg_cpu_usage: float = Field(0.0, description="Utilisation CPU moyenne (%)")
    avg_memory_usage: float = Field(0.0, description="Utilisation mémoire moyenne (%)")
    
    # Timestamp
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('performance_score', 'error_rate_percentage', 'availability_percentage', 
              'slo_compliance_percentage', 'avg_cpu_usage', 'avg_memory_usage')
    def validate_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Percentage must be between 0 and 100')
        return v


class PerformanceMonitoringService(BaseModel):
    """Service de monitoring de performance ultra-avancé"""
    
    # Configuration
    service_name: str = Field("performance-monitoring", description="Nom service")
    version: str = Field("1.0.0", description="Version")
    
    # Services surveillés
    monitored_services: List[str] = Field(default_factory=list, description="Services surveillés")
    
    # SLOs configurés
    slos: Dict[str, SLO] = Field(default_factory=dict, description="SLOs configurés")
    
    # Tests synthétiques
    synthetic_tests: Dict[str, SyntheticTest] = Field(
        default_factory=dict, description="Tests synthétiques"
    )
    
    # Configuration APM
    apm_enabled: bool = Field(True, description="APM activé")
    rum_enabled: bool = Field(True, description="RUM activé")
    profiling_enabled: bool = Field(True, description="Profiling activé")
    
    # Tracing distribué
    distributed_tracing: bool = Field(True, description="Tracing distribué")
    trace_sampling_rate: float = Field(0.1, description="Taux échantillonnage traces")
    
    # Base de données
    database_monitoring: bool = Field(True, description="Monitoring BDD")
    slow_query_threshold_ms: int = Field(1000, description="Seuil requêtes lentes (ms)")
    
    # Infrastructure
    infrastructure_monitoring: bool = Field(True, description="Monitoring infrastructure")
    
    # Business monitoring
    business_transaction_monitoring: bool = Field(True, description="Monitoring transactions business")
    
    # Alertes
    alerting_enabled: bool = Field(True, description="Alertes activées")
    alert_channels: List[str] = Field(default_factory=list, description="Canaux alertes")
    
    # Auto-scaling
    auto_scaling_enabled: bool = Field(True, description="Auto-scaling activé")
    
    def add_slo(self, slo: SLO) -> None:
        """Ajouter un SLO"""
        if slo.slo_id in self.slos:
            raise ValueError(f"SLO {slo.slo_id} already exists")
        self.slos[slo.slo_id] = slo
    
    def get_slo(self, slo_id: str) -> Optional[SLO]:
        """Récupérer un SLO"""
        return self.slos.get(slo_id)
    
    def add_synthetic_test(self, test: SyntheticTest) -> None:
        """Ajouter un test synthétique"""
        if test.test_id in self.synthetic_tests:
            raise ValueError(f"Test {test.test_id} already exists")
        self.synthetic_tests[test.test_id] = test
    
    @validator('trace_sampling_rate')
    def validate_sampling_rate(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Sampling rate must be between 0 and 1')
        return v


# SLOs prédéfinis pour Spotify AI Agent
SPOTIFY_SLOS = [
    SLO(
        slo_id="spotify_api_availability",
        name="Spotify API Availability",
        description="Disponibilité des APIs Spotify critiques",
        service_name="spotify-api",
        service_tier=ServiceTier.CRITICAL,
        target_percentage=99.9,
        measurement_window="30d",
        sli_metric="availability",
        sli_query="sum(rate(http_requests_total{job='spotify-api',code!~'5..'}[5m])) / sum(rate(http_requests_total{job='spotify-api'}[5m]))",
        good_event_query="sum(rate(http_requests_total{job='spotify-api',code!~'5..'}[5m]))",
        total_event_query="sum(rate(http_requests_total{job='spotify-api'}[5m]))",
        owner="Platform Team",
        team="SRE"
    ),
    
    SLO(
        slo_id="spotify_api_latency",
        name="Spotify API Latency P95",
        description="Latence P95 des APIs Spotify sous 500ms",
        service_name="spotify-api",
        service_tier=ServiceTier.CRITICAL,
        target_percentage=95.0,
        measurement_window="30d",
        sli_metric="latency",
        sli_query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job='spotify-api'}[5m])) < 0.5",
        good_event_query="sum(rate(http_request_duration_seconds_bucket{job='spotify-api',le='0.5'}[5m]))",
        total_event_query="sum(rate(http_request_duration_seconds_count{job='spotify-api'}[5m]))",
        owner="Platform Team",
        team="SRE"
    )
]


def create_default_performance_monitoring_service() -> PerformanceMonitoringService:
    """Créer service de monitoring performance par défaut"""
    service = PerformanceMonitoringService(
        monitored_services=["spotify-api", "recommendation-service", "user-service"],
        alert_channels=["slack", "pagerduty", "email"],
        apm_enabled=True,
        rum_enabled=True,
        distributed_tracing=True,
        business_transaction_monitoring=True
    )
    
    # Ajouter SLOs prédéfinis
    for slo in SPOTIFY_SLOS:
        service.add_slo(slo)
    
    return service


# Export des classes principales
__all__ = [
    "ServiceTier",
    "PerformanceStatus",
    "TransactionType",
    "AlertThreshold",
    "SLO",
    "ApplicationMetrics",
    "DatabaseMetrics",
    "InfrastructureMetrics",
    "BusinessTransaction",
    "SyntheticTest",
    "PerformanceProfile",
    "PerformanceMonitoringService",
    "SPOTIFY_SLOS",
    "create_default_performance_monitoring_service"
]
