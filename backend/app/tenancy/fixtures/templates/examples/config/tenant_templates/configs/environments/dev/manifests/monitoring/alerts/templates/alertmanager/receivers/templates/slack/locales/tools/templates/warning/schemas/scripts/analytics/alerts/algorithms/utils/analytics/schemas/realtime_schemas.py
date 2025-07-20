"""
Real-time Event Schemas - Ultra-Advanced Edition
===============================================

Schémas ultra-avancés pour les événements temps réel avec streaming,
WebSockets, message queuing et traitement distribué.

Features:
- Streaming d'événements haute performance
- WebSocket events avec état de connexion
- Message queuing avec garanties de livraison
- Event sourcing et CQRS patterns
- Traitement distribué et parallèle
- Réplication et synchronisation
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Literal, Set, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, PositiveInt, NonNegativeFloat, PositiveFloat
import json


class StreamEventType(str, Enum):
    """Types d'événements de streaming."""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    BUSINESS_EVENT = "business_event"
    METRIC_UPDATE = "metric_update"
    ALERT_TRIGGER = "alert_trigger"
    STATE_CHANGE = "state_change"
    NOTIFICATION = "notification"
    HEALTH_CHECK = "health_check"
    AUDIT_EVENT = "audit_event"
    SYNC_EVENT = "sync_event"


class EventPriority(str, Enum):
    """Priorités des événements."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class DeliveryGuarantee(str, Enum):
    """Garanties de livraison."""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class ConnectionState(str, Enum):
    """États de connexion WebSocket."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class StreamEvent(BaseModel):
    """Événement de streaming ultra-avancé."""
    
    # Identification unique
    event_id: UUID4 = Field(default_factory=lambda: UUID4())
    correlation_id: Optional[str] = Field(None, description="ID de corrélation")
    causation_id: Optional[UUID4] = Field(None, description="ID d'événement causateur")
    
    # Métadonnées temporelles
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(None, description="Timestamp de traitement")
    
    # Classification
    event_type: StreamEventType = Field(..., description="Type d'événement")
    event_name: str = Field(..., min_length=1, max_length=100, description="Nom de l'événement")
    event_version: str = Field(default="1.0.0", description="Version du schéma d'événement")
    
    # Source et contexte
    source_service: str = Field(..., description="Service source")
    source_instance: Optional[str] = Field(None, description="Instance source")
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    user_id: Optional[UUID4] = Field(None, description="ID utilisateur")
    session_id: Optional[str] = Field(None, description="ID de session")
    
    # Contenu de l'événement
    data: Dict[str, Any] = Field(..., description="Données de l'événement")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées")
    
    # Priorité et traitement
    priority: EventPriority = Field(default=EventPriority.NORMAL)
    delivery_guarantee: DeliveryGuarantee = Field(default=DeliveryGuarantee.AT_LEAST_ONCE)
    
    # Traçabilité
    trace_id: Optional[str] = Field(None, description="ID de trace distribuée")
    span_id: Optional[str] = Field(None, description="ID de span")
    parent_span_id: Optional[str] = Field(None, description="ID de span parent")
    
    # Configuration de routage
    routing_key: Optional[str] = Field(None, description="Clé de routage")
    topic: Optional[str] = Field(None, description="Topic de destination")
    partition_key: Optional[str] = Field(None, description="Clé de partitionnement")
    
    # TTL et expiration
    ttl_seconds: Optional[PositiveInt] = Field(None, description="TTL en secondes")
    expires_at: Optional[datetime] = Field(None, description="Date d'expiration")
    
    # Retry et erreurs
    retry_count: int = Field(default=0, ge=0, description="Nombre de tentatives")
    max_retries: int = Field(default=3, ge=0, description="Nombre max de tentatives")
    last_error: Optional[str] = Field(None, description="Dernière erreur")
    
    # Statut de traitement
    processed: bool = Field(default=False, description="Événement traité")
    processing_duration_ms: Optional[NonNegativeFloat] = Field(None, description="Durée de traitement")
    
    @validator('data')
    def validate_data_size(cls, v):
        # Limite la taille des données à 1MB pour les événements streaming
        data_str = json.dumps(v, default=str)
        if len(data_str.encode('utf-8')) > 1024 * 1024:
            raise ValueError("Event data size exceeds 1MB limit")
        return v
    
    @validator('expires_at')
    def validate_expiration(cls, v, values):
        if v is not None and 'timestamp' in values:
            if v <= values['timestamp']:
                raise ValueError("Expiration time must be in the future")
        return v


class RealtimeMetrics(BaseModel):
    """Métriques temps réel avec agrégation et fenêtrage."""
    
    metric_id: UUID4 = Field(default_factory=lambda: UUID4())
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Identification
    metric_name: str = Field(..., description="Nom de la métrique")
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    source: str = Field(..., description="Source de la métrique")
    
    # Valeurs
    value: float = Field(..., description="Valeur de la métrique")
    previous_value: Optional[float] = Field(None, description="Valeur précédente")
    
    # Fenêtrage temporel
    window_size_seconds: PositiveInt = Field(default=60, description="Taille de fenêtre")
    window_start: datetime = Field(..., description="Début de fenêtre")
    window_end: datetime = Field(..., description="Fin de fenêtre")
    
    # Agrégations
    min_value: Optional[float] = Field(None, description="Valeur minimale")
    max_value: Optional[float] = Field(None, description="Valeur maximale")
    avg_value: Optional[float] = Field(None, description="Valeur moyenne")
    sum_value: Optional[float] = Field(None, description="Somme")
    count: Optional[int] = Field(None, description="Nombre d'échantillons")
    
    # Statistiques avancées
    percentile_50: Optional[float] = Field(None, description="Percentile 50")
    percentile_95: Optional[float] = Field(None, description="Percentile 95")
    percentile_99: Optional[float] = Field(None, description="Percentile 99")
    standard_deviation: Optional[float] = Field(None, description="Écart-type")
    
    # Tendances
    trend_direction: Optional[str] = Field(None, description="Direction de tendance")
    trend_strength: Optional[float] = Field(None, ge=0.0, le=1.0, description="Force de tendance")
    change_rate: Optional[float] = Field(None, description="Taux de changement")
    
    # Contexte
    dimensions: Dict[str, str] = Field(default_factory=dict, description="Dimensions")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags")
    
    # Qualité des données
    sample_count: int = Field(default=1, ge=1, description="Nombre d'échantillons")
    confidence_level: float = Field(default=1.0, ge=0.0, le=1.0, description="Niveau de confiance")
    is_estimated: bool = Field(default=False, description="Valeur estimée")
    
    @validator('window_end')
    def validate_window(cls, v, values):
        if 'window_start' in values and v <= values['window_start']:
            raise ValueError("Window end must be after window start")
        return v


class WebSocketEvent(BaseModel):
    """Événement WebSocket avec gestion de connexion avancée."""
    
    # Identification
    event_id: UUID4 = Field(default_factory=lambda: UUID4())
    connection_id: str = Field(..., description="ID de connexion WebSocket")
    session_id: Optional[str] = Field(None, description="ID de session")
    
    # Métadonnées temporelles
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Type d'événement WebSocket
    event_type: str = Field(..., description="Type d'événement WS")
    action: str = Field(..., description="Action demandée")
    
    # Contenu
    payload: Dict[str, Any] = Field(default_factory=dict, description="Charge utile")
    headers: Dict[str, str] = Field(default_factory=dict, description="En-têtes")
    
    # Connexion
    connection_state: ConnectionState = Field(..., description="État de connexion")
    client_ip: Optional[str] = Field(None, description="IP du client")
    user_agent: Optional[str] = Field(None, description="User agent")
    
    # Contexte utilisateur
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    user_id: Optional[UUID4] = Field(None, description="ID utilisateur")
    
    # Métadonnées de connexion
    connected_at: Optional[datetime] = Field(None, description="Heure de connexion")
    last_ping: Optional[datetime] = Field(None, description="Dernier ping")
    latency_ms: Optional[NonNegativeFloat] = Field(None, description="Latence en ms")
    
    # Gestion des erreurs
    error_code: Optional[str] = Field(None, description="Code d'erreur")
    error_message: Optional[str] = Field(None, description="Message d'erreur")
    
    # Métriques de performance
    processing_time_ms: Optional[NonNegativeFloat] = Field(None, description="Temps de traitement")
    queue_time_ms: Optional[NonNegativeFloat] = Field(None, description="Temps en file")
    
    # Configuration
    compression_enabled: bool = Field(default=False, description="Compression activée")
    binary_mode: bool = Field(default=False, description="Mode binaire")
    
    @validator('payload')
    def validate_payload_size(cls, v):
        # Limite pour les événements WebSocket
        payload_str = json.dumps(v, default=str)
        if len(payload_str.encode('utf-8')) > 64 * 1024:  # 64KB
            raise ValueError("WebSocket payload exceeds 64KB limit")
        return v


class EventBatch(BaseModel):
    """Lot d'événements pour traitement en lot optimisé."""
    
    batch_id: UUID4 = Field(default_factory=lambda: UUID4())
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Métadonnées du lot
    batch_size: PositiveInt = Field(..., description="Taille du lot")
    total_events: PositiveInt = Field(..., description="Nombre total d'événements")
    
    # Événements
    events: List[StreamEvent] = Field(..., description="Liste des événements")
    
    # Configuration de traitement
    processing_mode: str = Field(default="parallel", description="Mode de traitement")
    max_parallel_workers: PositiveInt = Field(default=10, description="Workers parallèles max")
    
    # Statut
    processed_count: int = Field(default=0, ge=0, description="Événements traités")
    failed_count: int = Field(default=0, ge=0, description="Événements échoués")
    
    # Métriques de performance
    processing_started_at: Optional[datetime] = Field(None, description="Début traitement")
    processing_completed_at: Optional[datetime] = Field(None, description="Fin traitement")
    total_processing_time_ms: Optional[NonNegativeFloat] = Field(None, description="Temps total")
    
    # Gestion des erreurs
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Erreurs rencontrées")
    retry_batch: bool = Field(default=False, description="Retry du lot complet")
    
    @validator('events')
    def validate_events_count(cls, v, values):
        if 'total_events' in values and len(v) != values['total_events']:
            raise ValueError("Events list size doesn't match total_events")
        return v
    
    @property
    def success_rate(self) -> float:
        """Calcule le taux de succès du lot."""
        if self.total_events == 0:
            return 0.0
        return self.processed_count / self.total_events
    
    @property
    def processing_duration_seconds(self) -> Optional[float]:
        """Calcule la durée de traitement en secondes."""
        if self.processing_started_at and self.processing_completed_at:
            return (self.processing_completed_at - self.processing_started_at).total_seconds()
        return None


class StreamingChannel(BaseModel):
    """Canal de streaming avec configuration avancée."""
    
    channel_id: UUID4 = Field(default_factory=lambda: UUID4())
    name: str = Field(..., min_length=1, max_length=100, description="Nom du canal")
    description: Optional[str] = Field(None, max_length=500, description="Description")
    
    # Configuration
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Paramètres de streaming
    buffer_size: PositiveInt = Field(default=1000, description="Taille du buffer")
    max_throughput_per_second: PositiveInt = Field(default=10000, description="Débit max/sec")
    retention_hours: PositiveInt = Field(default=24, description="Rétention en heures")
    
    # Partitionnement
    partition_count: PositiveInt = Field(default=1, description="Nombre de partitions")
    partition_strategy: str = Field(default="round_robin", description="Stratégie de partitionnement")
    
    # Durabilité
    replication_factor: PositiveInt = Field(default=1, description="Facteur de réplication")
    persistence_enabled: bool = Field(default=True, description="Persistance activée")
    
    # Sécurité
    encryption_enabled: bool = Field(default=True, description="Chiffrement activé")
    access_control_enabled: bool = Field(default=True, description="Contrôle d'accès activé")
    authorized_users: List[UUID4] = Field(default_factory=list, description="Utilisateurs autorisés")
    
    # Monitoring
    metrics_enabled: bool = Field(default=True, description="Métriques activées")
    alerting_enabled: bool = Field(default=True, description="Alerting activé")
    
    # Statut
    active: bool = Field(default=True, description="Canal actif")
    consumer_count: int = Field(default=0, ge=0, description="Nombre de consommateurs")
    producer_count: int = Field(default=0, ge=0, description="Nombre de producteurs")
    
    # Métriques de performance
    messages_per_second: NonNegativeFloat = Field(default=0.0, description="Messages/sec")
    bytes_per_second: NonNegativeFloat = Field(default=0.0, description="Bytes/sec")
    lag_seconds: NonNegativeFloat = Field(default=0.0, description="Lag en secondes")


class EventSubscription(BaseModel):
    """Abonnement à des événements avec filtrage avancé."""
    
    subscription_id: UUID4 = Field(default_factory=lambda: UUID4())
    name: str = Field(..., description="Nom de l'abonnement")
    
    # Configuration
    tenant_id: UUID4 = Field(..., description="ID du tenant")
    user_id: UUID4 = Field(..., description="ID utilisateur")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Filtres d'événements
    event_types: List[StreamEventType] = Field(..., description="Types d'événements")
    event_names: List[str] = Field(default_factory=list, description="Noms d'événements")
    source_services: List[str] = Field(default_factory=list, description="Services sources")
    
    # Filtres avancés
    filter_expression: Optional[str] = Field(None, description="Expression de filtre")
    metadata_filters: Dict[str, Any] = Field(default_factory=dict, description="Filtres métadonnées")
    
    # Configuration de livraison
    delivery_method: str = Field(default="webhook", description="Méthode de livraison")
    endpoint_url: Optional[str] = Field(None, description="URL de livraison")
    webhook_secret: Optional[str] = Field(None, description="Secret webhook")
    
    # Paramètres de performance
    batch_size: PositiveInt = Field(default=1, description="Taille de lot")
    max_delivery_delay_seconds: PositiveInt = Field(default=60, description="Délai max livraison")
    
    # Retry et durabilité
    max_retries: PositiveInt = Field(default=3, description="Tentatives max")
    retry_delay_seconds: PositiveInt = Field(default=30, description="Délai entre tentatives")
    dead_letter_enabled: bool = Field(default=True, description="Dead letter activé")
    
    # Statut
    active: bool = Field(default=True, description="Abonnement actif")
    last_delivered_at: Optional[datetime] = Field(None, description="Dernière livraison")
    
    # Métriques
    total_events_received: int = Field(default=0, ge=0, description="Événements reçus")
    successful_deliveries: int = Field(default=0, ge=0, description="Livraisons réussies")
    failed_deliveries: int = Field(default=0, ge=0, description="Livraisons échouées")
    
    @property
    def delivery_success_rate(self) -> float:
        """Calcule le taux de succès de livraison."""
        total = self.successful_deliveries + self.failed_deliveries
        if total == 0:
            return 0.0
        return self.successful_deliveries / total


# Export des classes principales
__all__ = [
    "StreamEventType",
    "EventPriority",
    "DeliveryGuarantee",
    "ConnectionState",
    "StreamEvent",
    "RealtimeMetrics",
    "WebSocketEvent",
    "EventBatch",
    "StreamingChannel",
    "EventSubscription"
]
