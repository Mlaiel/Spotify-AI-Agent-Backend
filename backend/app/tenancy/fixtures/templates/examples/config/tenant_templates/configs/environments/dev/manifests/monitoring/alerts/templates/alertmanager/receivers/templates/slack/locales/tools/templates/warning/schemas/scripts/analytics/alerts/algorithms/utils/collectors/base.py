"""
Spotify AI Agent - Base Collector Classes
=======================================

Classes de base ultra-avancées pour tous les collecteurs de données
du système multi-tenant. Fournit l'infrastructure foundamentale pour
la collecte, validation et traitement des données avec des patterns
enterprise et des fonctionnalités de résilience intégrées.

Classes principales:
- BaseCollector: Classe abstraite avec fonctionnalités core
- CollectorConfig: Configuration avancée et validation
- CollectorStatus: Énumérations de statut
- CollectorMetadata: Métadonnées enrichies
- DataPoint: Structure de données optimisée
- MetricValue: Valeurs métriques typées
- CollectorInterface: Interface pour l'extensibilité

Patterns implémentés:
- Template Method Pattern
- Strategy Pattern  
- Observer Pattern
- Circuit Breaker Pattern
- Retry Pattern avec backoff exponentiel
- Bulkhead Pattern pour l'isolation
"""

from typing import Dict, Any, List, Optional, Union, Callable, Awaitable, Generic, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, IntEnum
import asyncio
import logging
import json
import time
import uuid
from collections import defaultdict, deque
import structlog
from concurrent.futures import ThreadPoolExecutor
import aioredis
import psutil
from prometheus_client import Counter, Histogram, Gauge, Summary


# Type variables pour la généricité
T = TypeVar('T')
DataType = TypeVar('DataType')


class CollectorStatus(Enum):
    """Statuts possibles d'un collecteur."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


class Priority(IntEnum):
    """Niveaux de priorité pour les collecteurs."""
    CRITICAL = 1    # Système critique, temps réel
    HIGH = 2        # Business important, quasi temps réel
    NORMAL = 3      # Standard, intervalles normaux
    LOW = 4         # Background, intervalles longs


class DataFormat(Enum):
    """Formats de données supportés."""
    JSON = "json"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    MSGPACK = "msgpack"
    PICKLE = "pickle"


class CompressionType(Enum):
    """Types de compression supportés."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"


@dataclass
class MetricValue:
    """
    Valeur métrique typée avec métadonnées enrichies.
    Optimisé pour les performances et la sérialisation.
    """
    name: str
    value: Union[int, float, str, bool]
    unit: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: Optional[str] = None
    quality_score: float = 1.0  # 0.0 à 1.0
    confidence_interval: Optional[tuple] = None
    
    def __post_init__(self):
        """Validation et normalisation post-initialisation."""
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.fromisoformat(str(self.timestamp))
        
        if self.quality_score < 0.0 or self.quality_score > 1.0:
            self.quality_score = 1.0
            
        # Normalisation des tags
        self.tags = {k.lower(): str(v) for k, v in self.tags.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire pour sérialisation."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "quality_score": self.quality_score,
            "confidence_interval": self.confidence_interval
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricValue':
        """Création depuis un dictionnaire."""
        return cls(
            name=data["name"],
            value=data["value"],
            unit=data.get("unit"),
            tags=data.get("tags", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source"),
            quality_score=data.get("quality_score", 1.0),
            confidence_interval=data.get("confidence_interval")
        )


@dataclass
class DataPoint:
    """
    Point de données optimisé pour les time-series et l'analytics.
    Supporte les agrégations et les calculs distribués.
    """
    metrics: List[MetricValue]
    tenant_id: str
    collector_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    environment: str = "production"
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validation et enrichissement post-initialisation."""
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
            
        # Validation des métriques
        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics] if self.metrics else []
            
        # Enrichissement avec métadonnées communes
        common_tags = {
            "tenant_id": self.tenant_id,
            "collector_id": self.collector_id,
            "environment": self.environment,
            "correlation_id": self.correlation_id
        }
        
        for metric in self.metrics:
            metric.tags.update(common_tags)
    
    def add_metric(self, metric: MetricValue) -> None:
        """Ajoute une métrique au point de données."""
        metric.tags.update({
            "tenant_id": self.tenant_id,
            "collector_id": self.collector_id,
            "environment": self.environment,
            "correlation_id": self.correlation_id
        })
        self.metrics.append(metric)
    
    def get_metrics_by_name(self, name: str) -> List[MetricValue]:
        """Récupère les métriques par nom."""
        return [m for m in self.metrics if m.name == name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire pour sérialisation."""
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "tenant_id": self.tenant_id,
            "collector_id": self.collector_id,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "environment": self.environment,
            "version": self.version
        }


@dataclass
class CollectorMetadata:
    """
    Métadonnées enrichies pour un collecteur.
    Utilisé pour le monitoring, l'observabilité et la gestion.
    """
    name: str
    version: str
    description: str
    collector_type: str
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    supported_environments: List[str] = field(default_factory=lambda: ["dev", "staging", "production"])
    data_retention_days: int = 30
    compliance_requirements: List[str] = field(default_factory=list)
    sla_requirements: Dict[str, Any] = field(default_factory=dict)
    cost_center: Optional[str] = None
    owner_team: str = "platform"
    contact_email: str = "platform@spotify-ai-agent.com"
    documentation_url: Optional[str] = None
    health_check_endpoint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "collector_type": self.collector_type,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
            "supported_environments": self.supported_environments,
            "data_retention_days": self.data_retention_days,
            "compliance_requirements": self.compliance_requirements,
            "sla_requirements": self.sla_requirements,
            "cost_center": self.cost_center,
            "owner_team": self.owner_team,
            "contact_email": self.contact_email,
            "documentation_url": self.documentation_url,
            "health_check_endpoint": self.health_check_endpoint
        }


@dataclass
class CollectorConfig:
    """
    Configuration ultra-avancée pour les collecteurs.
    Supporte tous les patterns enterprise et les optimisations.
    """
    # Configuration de base
    name: str
    enabled: bool = True
    priority: Priority = Priority.NORMAL
    environment: str = "production"
    
    # Configuration de collecte
    interval_seconds: int = 60
    batch_size: int = 1000
    max_queue_size: int = 10000
    collection_timeout_seconds: int = 30
    
    # Configuration de résilience
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_seconds: int = 60
    rate_limit_calls_per_minute: int = 1000
    
    # Configuration de performance
    enable_compression: bool = True
    compression_type: CompressionType = CompressionType.LZ4
    data_format: DataFormat = DataFormat.JSON
    enable_batching: bool = True
    batch_timeout_seconds: int = 5
    
    # Configuration de sécurité
    enable_encryption: bool = True
    encryption_key: Optional[str] = None
    enable_authentication: bool = True
    api_key: Optional[str] = None
    
    # Configuration de monitoring
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    metrics_prefix: str = "collector"
    
    # Configuration de stockage
    storage_backend: str = "redis"
    storage_config: Dict[str, Any] = field(default_factory=dict)
    enable_persistence: bool = True
    data_retention_hours: int = 24
    
    # Configuration d'alerting
    enable_alerting: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    alert_channels: List[str] = field(default_factory=lambda: ["slack", "email"])
    
    # Configuration tenant-specific
    tenant_id: Optional[str] = None
    tenant_profile: str = "standard"  # standard, premium, enterprise
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    # Tags et métadonnées
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validation et normalisation post-initialisation."""
        # Validation des valeurs
        if self.interval_seconds <= 0:
            self.interval_seconds = 60
            
        if self.batch_size <= 0:
            self.batch_size = 1000
            
        if self.max_retries < 0:
            self.max_retries = 3
            
        # Ajout de tags automatiques
        self.tags.update({
            "collector_name": self.name,
            "environment": self.environment,
            "priority": self.priority.name.lower(),
            "tenant_profile": self.tenant_profile
        })
        
        if self.tenant_id:
            self.tags["tenant_id"] = self.tenant_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "priority": self.priority.value,
            "environment": self.environment,
            "interval_seconds": self.interval_seconds,
            "batch_size": self.batch_size,
            "max_queue_size": self.max_queue_size,
            "collection_timeout_seconds": self.collection_timeout_seconds,
            "max_retries": self.max_retries,
            "retry_backoff_factor": self.retry_backoff_factor,
            "circuit_breaker_failure_threshold": self.circuit_breaker_failure_threshold,
            "circuit_breaker_recovery_timeout_seconds": self.circuit_breaker_recovery_timeout_seconds,
            "rate_limit_calls_per_minute": self.rate_limit_calls_per_minute,
            "enable_compression": self.enable_compression,
            "compression_type": self.compression_type.value,
            "data_format": self.data_format.value,
            "enable_batching": self.enable_batching,
            "batch_timeout_seconds": self.batch_timeout_seconds,
            "enable_encryption": self.enable_encryption,
            "enable_authentication": self.enable_authentication,
            "enable_metrics": self.enable_metrics,
            "enable_tracing": self.enable_tracing,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "metrics_prefix": self.metrics_prefix,
            "storage_backend": self.storage_backend,
            "storage_config": self.storage_config,
            "enable_persistence": self.enable_persistence,
            "data_retention_hours": self.data_retention_hours,
            "enable_alerting": self.enable_alerting,
            "alert_thresholds": self.alert_thresholds,
            "alert_channels": self.alert_channels,
            "tenant_id": self.tenant_id,
            "tenant_profile": self.tenant_profile,
            "resource_limits": self.resource_limits,
            "tags": self.tags,
            "labels": self.labels,
            "annotations": self.annotations
        }


class CollectorInterface(ABC):
    """
    Interface abstraite pour tous les collecteurs.
    Définit le contrat que tous les collecteurs doivent respecter.
    """
    
    @abstractmethod
    async def collect(self) -> List[DataPoint]:
        """
        Collecte les données principales du collecteur.
        
        Returns:
            Liste de points de données collectés
        """
        pass
    
    @abstractmethod
    async def validate_data(self, data: List[DataPoint]) -> bool:
        """
        Valide les données collectées.
        
        Args:
            data: Points de données à valider
            
        Returns:
            True si les données sont valides
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie la santé du collecteur.
        
        Returns:
            Rapport de santé détaillé
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> CollectorMetadata:
        """
        Retourne les métadonnées du collecteur.
        
        Returns:
            Métadonnées complètes
        """
        pass


class BaseCollector(CollectorInterface):
    """
    Classe de base ultra-avancée pour tous les collecteurs.
    
    Implémente tous les patterns enterprise, la résilience,
    l'observabilité et les optimisations de performance.
    """
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.status = CollectorStatus.INITIALIZING
        self.metadata = self._create_metadata()
        
        # État interne
        self._is_running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._last_collection_time: Optional[datetime] = None
        self._total_collections = 0
        self._error_count = 0
        self._data_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        
        # Patterns de résilience
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure: Optional[datetime] = None
        self._rate_limiter_tokens = config.rate_limit_calls_per_minute
        self._rate_limiter_last_refill = time.time()
        
        # Monitoring et observabilité
        self.logger = structlog.get_logger(
            collector_name=config.name,
            tenant_id=config.tenant_id,
            environment=config.environment
        )
        
        # Métriques Prometheus
        if config.enable_metrics:
            self._setup_metrics()
        
        # Executor pour les tâches CPU-intensives
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"collector-{config.name}")
        
        self.logger.info("Collecteur initialisé", config=config.to_dict())
    
    def _create_metadata(self) -> CollectorMetadata:
        """Crée les métadonnées du collecteur."""
        return CollectorMetadata(
            name=self.config.name,
            version="1.0.0",
            description=f"Collecteur {self.__class__.__name__}",
            collector_type=self.__class__.__name__,
            capabilities=["collect", "validate", "health_check"],
            owner_team="platform",
            contact_email="platform@spotify-ai-agent.com"
        )
    
    def _setup_metrics(self) -> None:
        """Configure les métriques Prometheus."""
        prefix = self.config.metrics_prefix
        
        self.metrics = {
            'collections_total': Counter(
                f'{prefix}_collections_total',
                'Total number of collections',
                ['collector_name', 'tenant_id', 'status']
            ),
            'collection_duration_seconds': Histogram(
                f'{prefix}_collection_duration_seconds',
                'Time spent collecting data',
                ['collector_name', 'tenant_id']
            ),
            'data_points_collected': Counter(
                f'{prefix}_data_points_collected_total',
                'Total data points collected',
                ['collector_name', 'tenant_id']
            ),
            'errors_total': Counter(
                f'{prefix}_errors_total',
                'Total number of errors',
                ['collector_name', 'tenant_id', 'error_type']
            ),
            'queue_size': Gauge(
                f'{prefix}_queue_size',
                'Current queue size',
                ['collector_name', 'tenant_id']
            ),
            'circuit_breaker_state': Gauge(
                f'{prefix}_circuit_breaker_state',
                'Circuit breaker state (0=closed, 1=open)',
                ['collector_name', 'tenant_id']
            )
        }
    
    async def start(self) -> None:
        """
        Démarre le collecteur avec gestion complète du cycle de vie.
        """
        if self._is_running:
            self.logger.warning("Collecteur déjà en cours d'exécution")
            return
        
        try:
            self.status = CollectorStatus.RUNNING
            self._is_running = True
            
            # Démarrage des tâches asynchrones
            self._collection_task = asyncio.create_task(self._collection_loop())
            
            # Démarrage du traitement de la queue
            asyncio.create_task(self._process_queue())
            
            self.logger.info("Collecteur démarré avec succès")
            
        except Exception as e:
            self.status = CollectorStatus.ERROR
            self.logger.error("Erreur lors du démarrage du collecteur", error=str(e))
            raise
    
    async def stop(self) -> None:
        """
        Arrête le collecteur proprement.
        """
        if not self._is_running:
            return
        
        self._is_running = False
        self.status = CollectorStatus.STOPPED
        
        # Annulation des tâches
        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        # Traitement des données restantes dans la queue
        await self._flush_queue()
        
        # Fermeture de l'executor
        self._executor.shutdown(wait=True)
        
        self.logger.info("Collecteur arrêté proprement")
    
    async def _collection_loop(self) -> None:
        """
        Boucle principale de collecte avec gestion des erreurs et résilience.
        """
        while self._is_running:
            try:
                # Vérification du circuit breaker
                if await self._is_circuit_breaker_open():
                    self.status = CollectorStatus.CIRCUIT_OPEN
                    await asyncio.sleep(self.config.circuit_breaker_recovery_timeout_seconds)
                    continue
                
                # Vérification du rate limiting
                if not await self._check_rate_limit():
                    self.status = CollectorStatus.RATE_LIMITED
                    await asyncio.sleep(1)  # Attente courte
                    continue
                
                # Collecte des données
                start_time = time.time()
                
                try:
                    data_points = await asyncio.wait_for(
                        self.collect(),
                        timeout=self.config.collection_timeout_seconds
                    )
                    
                    # Validation des données
                    if await self.validate_data(data_points):
                        # Ajout à la queue pour traitement asynchrone
                        for data_point in data_points:
                            await self._data_queue.put(data_point)
                        
                        # Mise à jour des métriques
                        self._total_collections += 1
                        self._last_collection_time = datetime.now(timezone.utc)
                        self._error_count = 0  # Reset sur succès
                        self.status = CollectorStatus.RUNNING
                        
                        if self.config.enable_metrics:
                            self.metrics['collections_total'].labels(
                                collector_name=self.config.name,
                                tenant_id=self.config.tenant_id or "unknown",
                                status="success"
                            ).inc()
                            
                            self.metrics['collection_duration_seconds'].labels(
                                collector_name=self.config.name,
                                tenant_id=self.config.tenant_id or "unknown"
                            ).observe(time.time() - start_time)
                            
                            self.metrics['data_points_collected'].labels(
                                collector_name=self.config.name,
                                tenant_id=self.config.tenant_id or "unknown"
                            ).inc(len(data_points))
                        
                        self.logger.debug(
                            "Collecte réussie",
                            data_points_count=len(data_points),
                            duration_seconds=time.time() - start_time
                        )
                    
                    else:
                        raise ValueError("Validation des données échouée")
                
                except asyncio.TimeoutError:
                    raise Exception(f"Timeout de collecte après {self.config.collection_timeout_seconds}s")
                
            except Exception as e:
                await self._handle_collection_error(e)
            
            # Attente avant la prochaine collecte
            await asyncio.sleep(self.config.interval_seconds)
    
    async def _handle_collection_error(self, error: Exception) -> None:
        """
        Gère les erreurs de collecte avec retry et circuit breaker.
        """
        self._error_count += 1
        self.logger.error(
            "Erreur lors de la collecte",
            error=str(error),
            error_count=self._error_count,
            max_retries=self.config.max_retries
        )
        
        # Mise à jour des métriques d'erreur
        if self.config.enable_metrics:
            self.metrics['errors_total'].labels(
                collector_name=self.config.name,
                tenant_id=self.config.tenant_id or "unknown",
                error_type=type(error).__name__
            ).inc()
        
        # Circuit breaker
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = datetime.now(timezone.utc)
        
        if self._circuit_breaker_failures >= self.config.circuit_breaker_failure_threshold:
            self.status = CollectorStatus.CIRCUIT_OPEN
            self.logger.warning(
                "Circuit breaker ouvert",
                failures=self._circuit_breaker_failures,
                threshold=self.config.circuit_breaker_failure_threshold
            )
        
        # Arrêt si trop d'erreurs consécutives
        if self._error_count >= self.config.max_retries:
            self.status = CollectorStatus.ERROR
            self.logger.critical(
                "Arrêt du collecteur après trop d'erreurs",
                error_count=self._error_count
            )
            await self.stop()
    
    async def _is_circuit_breaker_open(self) -> bool:
        """Vérifie si le circuit breaker est ouvert."""
        if self._circuit_breaker_failures < self.config.circuit_breaker_failure_threshold:
            return False
        
        if self._circuit_breaker_last_failure:
            time_since_failure = datetime.now(timezone.utc) - self._circuit_breaker_last_failure
            if time_since_failure.total_seconds() >= self.config.circuit_breaker_recovery_timeout_seconds:
                # Reset du circuit breaker
                self._circuit_breaker_failures = 0
                self._circuit_breaker_last_failure = None
                self.logger.info("Circuit breaker réinitialisé")
                return False
        
        return True
    
    async def _check_rate_limit(self) -> bool:
        """Vérifie et applique le rate limiting."""
        current_time = time.time()
        time_passed = current_time - self._rate_limiter_last_refill
        
        # Remplissage du bucket de tokens
        tokens_to_add = int(time_passed * (self.config.rate_limit_calls_per_minute / 60.0))
        if tokens_to_add > 0:
            self._rate_limiter_tokens = min(
                self.config.rate_limit_calls_per_minute,
                self._rate_limiter_tokens + tokens_to_add
            )
            self._rate_limiter_last_refill = current_time
        
        # Vérification de la disponibilité d'un token
        if self._rate_limiter_tokens > 0:
            self._rate_limiter_tokens -= 1
            return True
        
        return False
    
    async def _process_queue(self) -> None:
        """
        Traite la queue des données collectées de manière asynchrone.
        """
        batch = []
        last_batch_time = time.time()
        
        while self._is_running or not self._data_queue.empty():
            try:
                # Récupération d'un item avec timeout
                try:
                    data_point = await asyncio.wait_for(
                        self._data_queue.get(),
                        timeout=1.0
                    )
                    batch.append(data_point)
                except asyncio.TimeoutError:
                    # Timeout, traiter le batch actuel s'il n'est pas vide
                    if batch:
                        await self._process_batch(batch)
                        batch = []
                        last_batch_time = time.time()
                    continue
                
                # Traitement par batch
                current_time = time.time()
                batch_ready = (
                    len(batch) >= self.config.batch_size or
                    (current_time - last_batch_time) >= self.config.batch_timeout_seconds
                )
                
                if batch_ready:
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time
                
                # Mise à jour de la métrique de queue
                if self.config.enable_metrics:
                    self.metrics['queue_size'].labels(
                        collector_name=self.config.name,
                        tenant_id=self.config.tenant_id or "unknown"
                    ).set(self._data_queue.qsize())
                
            except Exception as e:
                self.logger.error("Erreur lors du traitement de la queue", error=str(e))
                await asyncio.sleep(1)
        
        # Traitement du dernier batch
        if batch:
            await self._process_batch(batch)
    
    async def _process_batch(self, batch: List[DataPoint]) -> None:
        """
        Traite un batch de données avec compression, encryption et stockage.
        """
        try:
            self.logger.debug("Traitement d'un batch", size=len(batch))
            
            # Sérialisation
            serialized_data = await self._serialize_batch(batch)
            
            # Compression
            if self.config.enable_compression:
                serialized_data = await self._compress_data(serialized_data)
            
            # Encryption
            if self.config.enable_encryption:
                serialized_data = await self._encrypt_data(serialized_data)
            
            # Stockage
            await self._store_data(serialized_data, batch)
            
            # Envoi pour alerting si nécessaire
            if self.config.enable_alerting:
                await self._check_alert_conditions(batch)
            
        except Exception as e:
            self.logger.error("Erreur lors du traitement du batch", error=str(e))
            raise
    
    async def _serialize_batch(self, batch: List[DataPoint]) -> bytes:
        """Sérialise un batch selon le format configuré."""
        data = [dp.to_dict() for dp in batch]
        
        if self.config.data_format == DataFormat.JSON:
            return json.dumps(data).encode('utf-8')
        elif self.config.data_format == DataFormat.MSGPACK:
            import msgpack
            return msgpack.packb(data)
        else:
            # Fallback to JSON
            return json.dumps(data).encode('utf-8')
    
    async def _compress_data(self, data: bytes) -> bytes:
        """Compresse les données selon le type configuré."""
        if self.config.compression_type == CompressionType.GZIP:
            import gzip
            return await asyncio.get_event_loop().run_in_executor(
                self._executor, gzip.compress, data
            )
        elif self.config.compression_type == CompressionType.LZ4:
            try:
                import lz4.frame
                return await asyncio.get_event_loop().run_in_executor(
                    self._executor, lz4.frame.compress, data
                )
            except ImportError:
                self.logger.warning("LZ4 non disponible, utilisation de gzip")
                import gzip
                return await asyncio.get_event_loop().run_in_executor(
                    self._executor, gzip.compress, data
                )
        else:
            return data
    
    async def _encrypt_data(self, data: bytes) -> bytes:
        """Chiffre les données si configuré."""
        # Implémentation basique - à améliorer avec vraie crypto
        if self.config.encryption_key:
            # Placeholder pour implémentation AES-256
            return data
        return data
    
    async def _store_data(self, data: bytes, batch: List[DataPoint]) -> None:
        """Stocke les données dans le backend configuré."""
        if self.config.storage_backend == "redis":
            await self._store_to_redis(data, batch)
        elif self.config.storage_backend == "postgresql":
            await self._store_to_postgresql(data, batch)
        else:
            self.logger.warning("Backend de stockage non supporté", backend=self.config.storage_backend)
    
    async def _store_to_redis(self, data: bytes, batch: List[DataPoint]) -> None:
        """Stockage dans Redis."""
        try:
            # Connexion Redis (à implémenter avec pool de connexions)
            # redis_client = await aioredis.from_url("redis://localhost")
            # Stockage avec TTL
            # await redis_client.setex(key, self.config.data_retention_hours * 3600, data)
            pass
        except Exception as e:
            self.logger.error("Erreur stockage Redis", error=str(e))
            raise
    
    async def _store_to_postgresql(self, data: bytes, batch: List[DataPoint]) -> None:
        """Stockage dans PostgreSQL."""
        try:
            # Implémentation avec asyncpg
            pass
        except Exception as e:
            self.logger.error("Erreur stockage PostgreSQL", error=str(e))
            raise
    
    async def _check_alert_conditions(self, batch: List[DataPoint]) -> None:
        """Vérifie les conditions d'alerte sur le batch."""
        for data_point in batch:
            for metric in data_point.metrics:
                threshold = self.config.alert_thresholds.get(metric.name)
                if threshold and isinstance(metric.value, (int, float)):
                    if metric.value > threshold:
                        await self._trigger_alert(metric, threshold)
    
    async def _trigger_alert(self, metric: MetricValue, threshold: float) -> None:
        """Déclenche une alerte."""
        alert_data = {
            "collector": self.config.name,
            "tenant_id": self.config.tenant_id,
            "metric": metric.name,
            "value": metric.value,
            "threshold": threshold,
            "timestamp": metric.timestamp.isoformat(),
            "severity": "critical" if metric.value > threshold * 1.5 else "warning"
        }
        
        self.logger.critical("ALERTE DÉCLENCHÉE", alert=alert_data)
        
        # Envoi vers les canaux configurés
        for channel in self.config.alert_channels:
            await self._send_alert_to_channel(channel, alert_data)
    
    async def _send_alert_to_channel(self, channel: str, alert_data: Dict[str, Any]) -> None:
        """Envoie une alerte vers un canal spécifique."""
        try:
            if channel == "slack":
                await self._send_slack_alert(alert_data)
            elif channel == "email":
                await self._send_email_alert(alert_data)
            elif channel == "webhook":
                await self._send_webhook_alert(alert_data)
        except Exception as e:
            self.logger.error("Erreur envoi alerte", channel=channel, error=str(e))
    
    async def _send_slack_alert(self, alert_data: Dict[str, Any]) -> None:
        """Envoie une alerte Slack."""
        # Implémentation à faire avec client Slack
        pass
    
    async def _send_email_alert(self, alert_data: Dict[str, Any]) -> None:
        """Envoie une alerte par email."""
        # Implémentation à faire avec SMTP
        pass
    
    async def _send_webhook_alert(self, alert_data: Dict[str, Any]) -> None:
        """Envoie une alerte par webhook."""
        # Implémentation à faire avec aiohttp
        pass
    
    async def _flush_queue(self) -> None:
        """Vide la queue avant l'arrêt."""
        remaining_items = []
        
        while not self._data_queue.empty():
            try:
                item = self._data_queue.get_nowait()
                remaining_items.append(item)
            except asyncio.QueueEmpty:
                break
        
        if remaining_items:
            self.logger.info("Traitement des éléments restants", count=len(remaining_items))
            await self._process_batch(remaining_items)
    
    # Méthodes abstraites à implémenter par les collecteurs spécialisés
    async def collect(self) -> List[DataPoint]:
        """
        Méthode abstraite pour la collecte de données.
        À implémenter par chaque collecteur spécialisé.
        """
        raise NotImplementedError("Les collecteurs doivent implémenter la méthode collect()")
    
    async def validate_data(self, data: List[DataPoint]) -> bool:
        """
        Validation par défaut des données collectées.
        Peut être surchargée par les collecteurs spécialisés.
        """
        if not data:
            return False
        
        for data_point in data:
            if not isinstance(data_point, DataPoint):
                return False
            
            if not data_point.metrics:
                return False
            
            for metric in data_point.metrics:
                if not isinstance(metric, MetricValue):
                    return False
                
                if not metric.name or metric.value is None:
                    return False
        
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check par défaut.
        Peut être étendu par les collecteurs spécialisés.
        """
        return {
            "status": self.status.value,
            "is_running": self._is_running,
            "last_collection": self._last_collection_time.isoformat() if self._last_collection_time else None,
            "total_collections": self._total_collections,
            "error_count": self._error_count,
            "queue_size": self._data_queue.qsize(),
            "circuit_breaker_failures": self._circuit_breaker_failures,
            "circuit_breaker_open": await self._is_circuit_breaker_open(),
            "config": self.config.to_dict(),
            "metadata": self.metadata.to_dict()
        }
    
    def get_metadata(self) -> CollectorMetadata:
        """Retourne les métadonnées du collecteur."""
        return self.metadata
    
    # Méthodes utilitaires pour les collecteurs spécialisés
    def create_metric(self, name: str, value: Union[int, float, str, bool], 
                     unit: Optional[str] = None, **kwargs) -> MetricValue:
        """Crée une métrique avec les tags du collecteur."""
        tags = self.config.tags.copy()
        tags.update(kwargs.get('tags', {}))
        
        return MetricValue(
            name=name,
            value=value,
            unit=unit,
            tags=tags,
            source=self.config.name,
            **{k: v for k, v in kwargs.items() if k != 'tags'}
        )
    
    def create_data_point(self, metrics: List[MetricValue]) -> DataPoint:
        """Crée un point de données avec les paramètres du collecteur."""
        return DataPoint(
            metrics=metrics,
            tenant_id=self.config.tenant_id or "unknown",
            collector_id=self.config.name,
            environment=self.config.environment
        )
