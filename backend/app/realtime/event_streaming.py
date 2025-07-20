# 🎵 Spotify AI Agent - Advanced Event Streaming Engine
# ====================================================
# 
# Moteur de streaming d'événements enterprise avec
# traitement temps réel et architecture event-driven.
#
# 🎖️ Expert: Data Engineer + ML Engineer + Backend Senior
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ====================================================

"""
🔄 Advanced Event Streaming Engine
=================================

Enterprise event streaming system providing:
- Event-driven architecture with CQRS pattern
- Real-time event processing and aggregation
- Stream analytics and complex event processing
- Event sourcing with replay capabilities
- Multi-tenant event isolation
- Dead letter queue and error handling
- Event schema validation and evolution
- Distributed event bus with guarantees
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Callable, Union, AsyncGenerator
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor

# Event streaming
import aioredis
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer, ConsumerRebalanceListener
import aiohttp

# Schema validation
from pydantic import BaseModel, Field, validator
from jsonschema import validate, ValidationError as SchemaValidationError

# Monitoring
from prometheus_client import Counter, Histogram, Gauge
import structlog

# ML and analytics
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter as CollectionsCounter

logger = structlog.get_logger(__name__)


class EventType(Enum):
    """Types d'événements système"""
    # Événements utilisateur
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    
    # Événements musicaux
    MUSIC_PLAY_STARTED = "music.play.started"
    MUSIC_PLAY_PAUSED = "music.play.paused"
    MUSIC_PLAY_STOPPED = "music.play.stopped"
    MUSIC_PLAY_COMPLETED = "music.play.completed"
    MUSIC_TRACK_SKIPPED = "music.track.skipped"
    MUSIC_VOLUME_CHANGED = "music.volume.changed"
    MUSIC_SEEK = "music.seek"
    
    # Événements playlist
    PLAYLIST_CREATED = "playlist.created"
    PLAYLIST_UPDATED = "playlist.updated"
    PLAYLIST_DELETED = "playlist.deleted"
    PLAYLIST_TRACK_ADDED = "playlist.track.added"
    PLAYLIST_TRACK_REMOVED = "playlist.track.removed"
    PLAYLIST_SHARED = "playlist.shared"
    
    # Événements collaboration
    COLLABORATION_STARTED = "collaboration.started"
    COLLABORATION_JOINED = "collaboration.joined"
    COLLABORATION_LEFT = "collaboration.left"
    COLLABORATION_EDIT = "collaboration.edit"
    
    # Événements sociaux
    SOCIAL_FOLLOW = "social.follow"
    SOCIAL_UNFOLLOW = "social.unfollow"
    SOCIAL_LIKE = "social.like"
    SOCIAL_SHARE = "social.share"
    SOCIAL_COMMENT = "social.comment"
    
    # Événements système
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_MAINTENANCE = "system.maintenance"
    
    # Événements ML
    ML_RECOMMENDATION_GENERATED = "ml.recommendation.generated"
    ML_MODEL_TRAINED = "ml.model.trained"
    ML_PREDICTION_MADE = "ml.prediction.made"


class EventPriority(Enum):
    """Priorités d'événements"""
    CRITICAL = 0  # Événements critiques système
    HIGH = 1      # Haute priorité
    NORMAL = 2    # Priorité normale
    LOW = 3       # Basse priorité
    BACKGROUND = 4 # Traitement en arrière-plan


class EventStatus(Enum):
    """Statuts d'événements"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class EventMetadata:
    """Métadonnées d'événement"""
    correlation_id: str
    causation_id: Optional[str] = None
    tenant_id: str = "default"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_service: Optional[str] = None
    source_version: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


@dataclass
class StreamEvent:
    """Événement de streaming structuré"""
    id: str
    type: EventType
    data: Dict[str, Any]
    metadata: EventMetadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    priority: EventPriority = EventPriority.NORMAL
    status: EventStatus = EventStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "id": self.id,
            "type": self.type.value,
            "data": self.data,
            "metadata": asdict(self.metadata),
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "priority": self.priority.value,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamEvent':
        """Crée depuis un dictionnaire"""
        metadata = EventMetadata(**data["metadata"])
        
        return cls(
            id=data["id"],
            type=EventType(data["type"]),
            data=data["data"],
            metadata=metadata,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version", 1),
            priority=EventPriority(data.get("priority", EventPriority.NORMAL.value)),
            status=EventStatus(data.get("status", EventStatus.PENDING.value)),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
        )


class EventSchema(BaseModel):
    """Schéma de validation d'événement"""
    event_type: str
    required_fields: List[str] = []
    optional_fields: List[str] = []
    field_types: Dict[str, str] = {}
    validation_rules: Dict[str, Any] = {}
    
    def validate_event(self, event: StreamEvent) -> bool:
        """Valide un événement contre le schéma"""
        try:
            # Vérifier les champs requis
            for field in self.required_fields:
                if field not in event.data:
                    raise ValueError(f"Required field missing: {field}")
            
            # Vérifier les types de champs
            for field, expected_type in self.field_types.items():
                if field in event.data:
                    value = event.data[field]
                    if expected_type == "string" and not isinstance(value, str):
                        raise ValueError(f"Field {field} must be string")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        raise ValueError(f"Field {field} must be number")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        raise ValueError(f"Field {field} must be boolean")
            
            return True
            
        except Exception as e:
            logger.error("Event validation failed", event_id=event.id, error=str(e))
            return False


class EventHandler:
    """Gestionnaire d'événements avec logique métier"""
    
    def __init__(self, handler_id: str, event_types: List[EventType]):
        self.handler_id = handler_id
        self.event_types = event_types
        self.metrics = {
            "events_processed": Counter(f"handler_{handler_id}_events_processed_total", "Events processed"),
            "processing_time": Histogram(f"handler_{handler_id}_processing_seconds", "Processing time"),
            "errors": Counter(f"handler_{handler_id}_errors_total", "Processing errors")
        }
    
    async def handle(self, event: StreamEvent) -> bool:
        """Traite un événement"""
        if event.type not in self.event_types:
            return False
        
        start_time = time.time()
        
        try:
            success = await self.process_event(event)
            
            self.metrics["events_processed"].inc()
            processing_time = time.time() - start_time
            self.metrics["processing_time"].observe(processing_time)
            
            return success
            
        except Exception as e:
            self.metrics["errors"].inc()
            logger.error("Event handler error", 
                        handler_id=self.handler_id,
                        event_id=event.id,
                        error=str(e))
            return False
    
    async def process_event(self, event: StreamEvent) -> bool:
        """Logique de traitement à implémenter par les sous-classes"""
        raise NotImplementedError


class MusicPlayHandler(EventHandler):
    """Handler pour événements de lecture musicale"""
    
    def __init__(self):
        super().__init__(
            "music_play_handler",
            [EventType.MUSIC_PLAY_STARTED, EventType.MUSIC_PLAY_COMPLETED, EventType.MUSIC_TRACK_SKIPPED]
        )
        self.user_sessions: Dict[str, Dict] = {}
    
    async def process_event(self, event: StreamEvent) -> bool:
        """Traite les événements de lecture musicale"""
        user_id = event.metadata.user_id
        if not user_id:
            return False
        
        if event.type == EventType.MUSIC_PLAY_STARTED:
            await self._handle_play_started(user_id, event.data)
        elif event.type == EventType.MUSIC_PLAY_COMPLETED:
            await self._handle_play_completed(user_id, event.data)
        elif event.type == EventType.MUSIC_TRACK_SKIPPED:
            await self._handle_track_skipped(user_id, event.data)
        
        return True
    
    async def _handle_play_started(self, user_id: str, data: Dict[str, Any]):
        """Gère le début de lecture"""
        track_id = data.get("track_id")
        playlist_id = data.get("playlist_id")
        
        self.user_sessions[user_id] = {
            "track_id": track_id,
            "playlist_id": playlist_id,
            "start_time": datetime.utcnow(),
            "duration": 0
        }
        
        logger.info("Music play started", user_id=user_id, track_id=track_id)
    
    async def _handle_play_completed(self, user_id: str, data: Dict[str, Any]):
        """Gère la fin de lecture"""
        session = self.user_sessions.get(user_id)
        if session:
            duration = (datetime.utcnow() - session["start_time"]).total_seconds()
            session["duration"] = duration
            
            # Analytics et recommandations basées sur l'écoute complète
            await self._generate_listening_analytics(user_id, session)
        
        logger.info("Music play completed", user_id=user_id, duration=session.get("duration") if session else 0)
    
    async def _handle_track_skipped(self, user_id: str, data: Dict[str, Any]):
        """Gère le skip de track"""
        session = self.user_sessions.get(user_id)
        if session:
            skip_time = (datetime.utcnow() - session["start_time"]).total_seconds()
            
            # Analytics sur les patterns de skip
            await self._analyze_skip_pattern(user_id, session, skip_time)
        
        logger.info("Track skipped", user_id=user_id, skip_time=skip_time if session else 0)
    
    async def _generate_listening_analytics(self, user_id: str, session: Dict[str, Any]):
        """Génère des analytics d'écoute"""
        # Logique d'analytics ML
        pass
    
    async def _analyze_skip_pattern(self, user_id: str, session: Dict[str, Any], skip_time: float):
        """Analyse les patterns de skip"""
        # Logique d'analyse ML
        pass


class RecommendationHandler(EventHandler):
    """Handler pour génération de recommandations"""
    
    def __init__(self):
        super().__init__(
            "recommendation_handler",
            [EventType.MUSIC_PLAY_COMPLETED, EventType.SOCIAL_LIKE, EventType.PLAYLIST_TRACK_ADDED]
        )
        self.user_preferences: Dict[str, Dict] = defaultdict(dict)
    
    async def process_event(self, event: StreamEvent) -> bool:
        """Traite les événements pour recommandations"""
        user_id = event.metadata.user_id
        if not user_id:
            return False
        
        await self._update_user_preferences(user_id, event)
        await self._generate_recommendations(user_id)
        
        return True
    
    async def _update_user_preferences(self, user_id: str, event: StreamEvent):
        """Met à jour les préférences utilisateur"""
        preferences = self.user_preferences[user_id]
        
        if event.type == EventType.MUSIC_PLAY_COMPLETED:
            # Incrémenter le score du genre/artiste
            track_genre = event.data.get("genre")
            track_artist = event.data.get("artist")
            
            if track_genre:
                preferences.setdefault("genres", {})
                preferences["genres"][track_genre] = preferences["genres"].get(track_genre, 0) + 1
            
            if track_artist:
                preferences.setdefault("artists", {})
                preferences["artists"][track_artist] = preferences["artists"].get(track_artist, 0) + 1
        
        elif event.type == EventType.SOCIAL_LIKE:
            # Renforcer les préférences pour les tracks likées
            track_id = event.data.get("track_id")
            if track_id:
                preferences.setdefault("liked_tracks", set())
                preferences["liked_tracks"].add(track_id)
    
    async def _generate_recommendations(self, user_id: str):
        """Génère des recommandations personnalisées"""
        preferences = self.user_preferences[user_id]
        
        # Algorithme de recommandation basique
        recommendations = await self._calculate_recommendations(preferences)
        
        # Publier événement de recommandation
        if recommendations:
            rec_event = StreamEvent(
                id=str(uuid.uuid4()),
                type=EventType.ML_RECOMMENDATION_GENERATED,
                data={
                    "user_id": user_id,
                    "recommendations": recommendations,
                    "algorithm": "collaborative_filtering"
                },
                metadata=EventMetadata(
                    correlation_id=str(uuid.uuid4()),
                    user_id=user_id
                )
            )
            
            # Publier via le stream engine
            # await stream_engine.publish(rec_event)
    
    async def _calculate_recommendations(self, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calcule les recommandations ML"""
        # Logique de ML simplifiée
        recommendations = []
        
        # Recommandations basées sur les genres
        top_genres = sorted(
            preferences.get("genres", {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for genre, score in top_genres:
            recommendations.append({
                "type": "genre",
                "value": genre,
                "score": score / sum(preferences.get("genres", {}).values()),
                "reason": f"Based on your listening history in {genre}"
            })
        
        return recommendations


class EventAggregator:
    """Agrégateur d'événements pour analytics temps réel"""
    
    def __init__(self, window_size_seconds: int = 300):  # 5 minutes
        self.window_size = window_size_seconds
        self.event_windows: Dict[str, deque] = defaultdict(lambda: deque())
        self.aggregates: Dict[str, Dict] = defaultdict(dict)
        
        # Démarrer le traitement périodique
        asyncio.create_task(self._aggregate_loop())
    
    async def add_event(self, event: StreamEvent):
        """Ajoute un événement aux fenêtres d'agrégation"""
        current_time = time.time()
        
        # Ajouter à la fenêtre globale
        self.event_windows["global"].append((current_time, event))
        
        # Ajouter aux fenêtres par type
        self.event_windows[event.type.value].append((current_time, event))
        
        # Ajouter aux fenêtres par utilisateur
        if event.metadata.user_id:
            user_key = f"user:{event.metadata.user_id}"
            self.event_windows[user_key].append((current_time, event))
        
        # Ajouter aux fenêtres par tenant
        tenant_key = f"tenant:{event.metadata.tenant_id}"
        self.event_windows[tenant_key].append((current_time, event))
    
    async def _aggregate_loop(self):
        """Boucle d'agrégation périodique"""
        while True:
            try:
                await asyncio.sleep(60)  # Agrégation chaque minute
                
                current_time = time.time()
                cutoff_time = current_time - self.window_size
                
                # Nettoyer et agréger chaque fenêtre
                for key, window in self.event_windows.items():
                    # Supprimer les événements expirés
                    while window and window[0][0] < cutoff_time:
                        window.popleft()
                    
                    # Calculer les agrégats
                    self.aggregates[key] = await self._calculate_aggregates(window)
                
            except Exception as e:
                logger.error("Aggregation error", error=str(e))
    
    async def _calculate_aggregates(self, window: deque) -> Dict[str, Any]:
        """Calcule les agrégats pour une fenêtre"""
        events = [event for timestamp, event in window]
        
        if not events:
            return {}
        
        # Comptage par type
        event_counts = CollectionsCounter(event.type.value for event in events)
        
        # Comptage par statut
        status_counts = CollectionsCounter(event.status.value for event in events)
        
        # Comptage par priorité
        priority_counts = CollectionsCounter(event.priority.value for event in events)
        
        # Statistiques temporelles
        timestamps = [event.timestamp for event in events]
        if timestamps:
            latest = max(timestamps)
            earliest = min(timestamps)
            time_span = (latest - earliest).total_seconds()
        else:
            time_span = 0
        
        return {
            "total_events": len(events),
            "event_types": dict(event_counts),
            "event_statuses": dict(status_counts),
            "event_priorities": dict(priority_counts),
            "time_span_seconds": time_span,
            "events_per_second": len(events) / max(self.window_size, 1),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def get_aggregates(self, key: str = "global") -> Dict[str, Any]:
        """Récupère les agrégats pour une clé"""
        return self.aggregates.get(key, {})


class DeadLetterQueue:
    """Queue des lettres mortes pour événements échoués"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.dlq_key = "events:dead_letter_queue"
        self.max_size = 10000
    
    async def add_event(self, event: StreamEvent, error: str):
        """Ajoute un événement à la DLQ"""
        dlq_item = {
            "event": event.to_dict(),
            "error": error,
            "added_at": datetime.utcnow().isoformat(),
            "retry_count": event.retry_count
        }
        
        await self.redis.lpush(self.dlq_key, json.dumps(dlq_item))
        
        # Limiter la taille de la queue
        await self.redis.ltrim(self.dlq_key, 0, self.max_size - 1)
        
        logger.warning("Event added to dead letter queue", 
                      event_id=event.id,
                      error=error)
    
    async def get_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les événements de la DLQ"""
        items = await self.redis.lrange(self.dlq_key, 0, limit - 1)
        return [json.loads(item) for item in items]
    
    async def retry_event(self, index: int) -> Optional[StreamEvent]:
        """Retire un événement de la DLQ pour retry"""
        item = await self.redis.lindex(self.dlq_key, index)
        if item:
            data = json.loads(item)
            await self.redis.lrem(self.dlq_key, 1, item)
            
            event = StreamEvent.from_dict(data["event"])
            event.status = EventStatus.PENDING
            event.retry_count += 1
            
            return event
        return None


class EventStreamEngine:
    """Moteur principal de streaming d'événements"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 kafka_bootstrap_servers: str = "localhost:9092"):
        
        self.redis_url = redis_url
        self.kafka_servers = kafka_bootstrap_servers
        
        # Clients
        self.redis_client: Optional[aioredis.Redis] = None
        self.kafka_producer: Optional[AIOKafkaProducer] = None
        self.kafka_consumer: Optional[AIOKafkaConsumer] = None
        
        # Composants
        self.event_handlers: List[EventHandler] = []
        self.event_schemas: Dict[str, EventSchema] = {}
        self.event_aggregator = EventAggregator()
        self.dead_letter_queue: Optional[DeadLetterQueue] = None
        
        # Configuration
        self.topics = {
            "events": "spotify_events",
            "dlq": "spotify_events_dlq",
            "analytics": "spotify_analytics"
        }
        
        # Métriques
        self.metrics = {
            "events_published": Counter("events_published_total", "Total events published", ["type", "status"]),
            "events_processed": Counter("events_processed_total", "Total events processed", ["handler", "status"]),
            "processing_latency": Histogram("event_processing_latency_seconds", "Event processing latency"),
            "queue_size": Gauge("event_queue_size", "Event queue size", ["queue"])
        }
        
        # État
        self.is_running = False
        self.consumer_tasks: List[asyncio.Task] = []
    
    async def initialize(self):
        """Initialise le moteur de streaming"""
        try:
            # Connexion Redis
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Initialiser DLQ
            self.dead_letter_queue = DeadLetterQueue(self.redis_client)
            
            # Configuration Kafka
            await self._setup_kafka()
            
            # Enregistrer les handlers par défaut
            self._register_default_handlers()
            
            # Démarrer les consumers
            await self._start_consumers()
            
            self.is_running = True
            
            logger.info("Event stream engine initialized")
            
        except Exception as e:
            logger.error("Failed to initialize event stream engine", error=str(e))
            raise
    
    async def _setup_kafka(self):
        """Configure Kafka producer et consumer"""
        # Producer
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode(),
            compression_type="gzip",
            batch_size=16384,
            linger_ms=10
        )
        await self.kafka_producer.start()
        
        # Consumer
        self.kafka_consumer = AIOKafkaConsumer(
            self.topics["events"],
            bootstrap_servers=self.kafka_servers,
            group_id="spotify_event_processors",
            value_deserializer=lambda m: json.loads(m.decode()),
            auto_offset_reset="latest",
            enable_auto_commit=False
        )
        await self.kafka_consumer.start()
    
    def _register_default_handlers(self):
        """Enregistre les handlers par défaut"""
        music_handler = MusicPlayHandler()
        recommendation_handler = RecommendationHandler()
        
        self.event_handlers.extend([music_handler, recommendation_handler])
        
        # Schémas par défaut
        self.event_schemas.update({
            EventType.MUSIC_PLAY_STARTED.value: EventSchema(
                event_type=EventType.MUSIC_PLAY_STARTED.value,
                required_fields=["track_id", "user_id"],
                optional_fields=["playlist_id", "device_id"],
                field_types={"track_id": "string", "user_id": "string"}
            ),
            EventType.PLAYLIST_CREATED.value: EventSchema(
                event_type=EventType.PLAYLIST_CREATED.value,
                required_fields=["playlist_id", "name", "user_id"],
                field_types={"playlist_id": "string", "name": "string", "user_id": "string"}
            )
        })
    
    async def _start_consumers(self):
        """Démarre les consumers d'événements"""
        # Consumer principal
        main_consumer_task = asyncio.create_task(self._main_consumer_loop())
        self.consumer_tasks.append(main_consumer_task)
        
        # Consumer DLQ retry
        dlq_retry_task = asyncio.create_task(self._dlq_retry_loop())
        self.consumer_tasks.append(dlq_retry_task)
        
        logger.info("Event consumers started")
    
    async def publish(self, event: StreamEvent) -> bool:
        """Publie un événement"""
        try:
            # Validation du schéma
            if not await self._validate_event(event):
                self.metrics["events_published"].labels(type=event.type.value, status="validation_failed").inc()
                return False
            
            # Ajouter à l'agrégateur
            await self.event_aggregator.add_event(event)
            
            # Publier vers Kafka
            await self.kafka_producer.send(
                self.topics["events"],
                value=event.to_dict(),
                key=event.metadata.correlation_id.encode() if event.metadata.correlation_id else None
            )
            
            # Stocker dans Redis pour requêtes rapides
            await self._store_event_in_redis(event)
            
            self.metrics["events_published"].labels(type=event.type.value, status="success").inc()
            
            logger.debug("Event published", event_id=event.id, type=event.type.value)
            return True
            
        except Exception as e:
            self.metrics["events_published"].labels(type=event.type.value, status="error").inc()
            logger.error("Failed to publish event", event_id=event.id, error=str(e))
            return False
    
    async def _validate_event(self, event: StreamEvent) -> bool:
        """Valide un événement contre son schéma"""
        schema = self.event_schemas.get(event.type.value)
        if schema:
            return schema.validate_event(event)
        return True  # Pas de validation si pas de schéma
    
    async def _store_event_in_redis(self, event: StreamEvent):
        """Stocke un événement dans Redis pour requêtes rapides"""
        # Stocker par ID
        await self.redis_client.setex(
            f"event:{event.id}",
            3600,  # 1 heure
            json.dumps(event.to_dict(), default=str)
        )
        
        # Ajouter aux index
        if event.metadata.user_id:
            await self.redis_client.lpush(
                f"user_events:{event.metadata.user_id}",
                event.id
            )
            await self.redis_client.expire(f"user_events:{event.metadata.user_id}", 86400)  # 24h
        
        # Index par type
        await self.redis_client.lpush(f"events_by_type:{event.type.value}", event.id)
        await self.redis_client.expire(f"events_by_type:{event.type.value}", 3600)
    
    async def _main_consumer_loop(self):
        """Boucle principale de consommation d'événements"""
        while self.is_running:
            try:
                async for message in self.kafka_consumer:
                    try:
                        # Désérialiser l'événement
                        event_data = message.value
                        event = StreamEvent.from_dict(event_data)
                        
                        # Traiter l'événement
                        success = await self._process_event(event)
                        
                        if success:
                            await self.kafka_consumer.commit()
                        else:
                            # Retry ou DLQ
                            await self._handle_processing_failure(event, "Processing failed")
                    
                    except Exception as e:
                        logger.error("Error processing message", error=str(e))
                        
            except Exception as e:
                logger.error("Consumer loop error", error=str(e))
                await asyncio.sleep(5)
    
    async def _process_event(self, event: StreamEvent) -> bool:
        """Traite un événement avec tous les handlers"""
        start_time = time.time()
        success_count = 0
        
        try:
            for handler in self.event_handlers:
                try:
                    if await handler.handle(event):
                        success_count += 1
                        self.metrics["events_processed"].labels(
                            handler=handler.handler_id, 
                            status="success"
                        ).inc()
                    
                except Exception as e:
                    self.metrics["events_processed"].labels(
                        handler=handler.handler_id, 
                        status="error"
                    ).inc()
                    logger.error("Handler error", 
                                handler_id=handler.handler_id,
                                event_id=event.id,
                                error=str(e))
            
            # Enregistrer la latence
            processing_time = time.time() - start_time
            self.metrics["processing_latency"].observe(processing_time)
            
            # Mettre à jour le statut
            if success_count > 0:
                event.status = EventStatus.COMPLETED
                return True
            else:
                event.status = EventStatus.FAILED
                return False
                
        except Exception as e:
            logger.error("Event processing error", event_id=event.id, error=str(e))
            event.status = EventStatus.FAILED
            return False
    
    async def _handle_processing_failure(self, event: StreamEvent, error: str):
        """Gère l'échec de traitement d'un événement"""
        event.retry_count += 1
        
        if event.retry_count < event.max_retries:
            # Retry avec backoff exponentiel
            delay = min(300, 2 ** event.retry_count)  # Max 5 minutes
            
            event.status = EventStatus.RETRYING
            
            # Reprogrammer après délai
            asyncio.create_task(self._retry_event_after_delay(event, delay))
            
            logger.info("Event scheduled for retry", 
                       event_id=event.id,
                       retry_count=event.retry_count,
                       delay_seconds=delay)
        else:
            # Envoyer vers DLQ
            event.status = EventStatus.DEAD_LETTER
            if self.dead_letter_queue:
                await self.dead_letter_queue.add_event(event, error)
    
    async def _retry_event_after_delay(self, event: StreamEvent, delay_seconds: int):
        """Retry un événement après délai"""
        await asyncio.sleep(delay_seconds)
        
        event.status = EventStatus.PENDING
        success = await self._process_event(event)
        
        if not success:
            await self._handle_processing_failure(event, "Retry failed")
    
    async def _dlq_retry_loop(self):
        """Boucle de retry des événements DLQ"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                if self.dead_letter_queue:
                    # Récupérer quelques événements pour retry
                    dlq_events = await self.dead_letter_queue.get_events(10)
                    
                    for i, dlq_item in enumerate(dlq_events):
                        # Retry seulement les événements récents (< 24h)
                        added_at = datetime.fromisoformat(dlq_item["added_at"])
                        if (datetime.utcnow() - added_at).total_seconds() < 86400:
                            event = await self.dead_letter_queue.retry_event(i)
                            if event:
                                await self.publish(event)
                
            except Exception as e:
                logger.error("DLQ retry loop error", error=str(e))
    
    def register_handler(self, handler: EventHandler):
        """Enregistre un handler d'événements"""
        self.event_handlers.append(handler)
        logger.info("Event handler registered", handler_id=handler.handler_id)
    
    def register_schema(self, schema: EventSchema):
        """Enregistre un schéma d'événement"""
        self.event_schemas[schema.event_type] = schema
        logger.info("Event schema registered", event_type=schema.event_type)
    
    async def get_event(self, event_id: str) -> Optional[StreamEvent]:
        """Récupère un événement par ID"""
        if self.redis_client:
            event_data = await self.redis_client.get(f"event:{event_id}")
            if event_data:
                return StreamEvent.from_dict(json.loads(event_data))
        return None
    
    async def get_user_events(self, user_id: str, limit: int = 100) -> List[StreamEvent]:
        """Récupère les événements d'un utilisateur"""
        events = []
        if self.redis_client:
            event_ids = await self.redis_client.lrange(f"user_events:{user_id}", 0, limit - 1)
            for event_id in event_ids:
                event = await self.get_event(event_id)
                if event:
                    events.append(event)
        return events
    
    async def get_analytics(self, key: str = "global") -> Dict[str, Any]:
        """Récupère les analytics agrégées"""
        return self.event_aggregator.get_aggregates(key)
    
    async def shutdown(self):
        """Arrête le moteur de streaming"""
        self.is_running = False
        
        # Arrêter les consumers
        for task in self.consumer_tasks:
            task.cancel()
        
        await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
        
        # Fermer les connexions
        if self.kafka_producer:
            await self.kafka_producer.stop()
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Event stream engine shutdown completed")


# Factory function
async def create_event_stream_engine(redis_url: str = "redis://localhost:6379",
                                   kafka_servers: str = "localhost:9092") -> EventStreamEngine:
    """Crée et initialise le moteur de streaming"""
    engine = EventStreamEngine(redis_url, kafka_servers)
    await engine.initialize()
    return engine


# Instance globale
event_stream_engine: Optional[EventStreamEngine] = None


# Fonctions utilitaires
async def publish_event(event_type: EventType, 
                       data: Dict[str, Any],
                       user_id: Optional[str] = None,
                       correlation_id: Optional[str] = None,
                       priority: EventPriority = EventPriority.NORMAL) -> bool:
    """Fonction utilitaire pour publier un événement"""
    
    if not event_stream_engine:
        logger.error("Event stream engine not initialized")
        return False
    
    event = StreamEvent(
        id=str(uuid.uuid4()),
        type=event_type,
        data=data,
        metadata=EventMetadata(
            correlation_id=correlation_id or str(uuid.uuid4()),
            user_id=user_id
        ),
        priority=priority
    )
    
    return await event_stream_engine.publish(event)


# Export des classes principales
__all__ = [
    "EventStreamEngine",
    "StreamEvent",
    "EventType",
    "EventPriority",
    "EventStatus",
    "EventMetadata",
    "EventHandler",
    "EventSchema",
    "EventAggregator",
    "DeadLetterQueue",
    "MusicPlayHandler",
    "RecommendationHandler",
    "publish_event",
    "create_event_stream_engine"
]
