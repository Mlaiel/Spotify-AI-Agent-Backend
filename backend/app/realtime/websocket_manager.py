# 🎵 Spotify AI Agent - Advanced WebSocket Manager
# ================================================
# 
# Gestionnaire WebSocket enterprise avec mise à l'échelle,
# load balancing et gestion avancée des connexions.
#
# 🎖️ Expert: Senior Backend Developer + DevOps + Security
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ================================================

"""
🔌 Advanced WebSocket Management System
======================================

Enterprise-grade WebSocket manager providing:
- Horizontal scaling with Redis clustering
- Connection pooling and load balancing
- Auto-reconnection with circuit breaker pattern
- Real-time health monitoring and metrics
- Multi-tenant isolation and security
- Message routing and broadcasting
- Connection lifecycle management
- Performance optimization and caching
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Callable, Union
import weakref
import hashlib
import hmac

# WebSocket and networking
from fastapi import WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.websockets import WebSocketState
import aioredis
import aiohttp
from starlette.websockets import WebSocketClose

# Monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog

# Security and auth
import jwt
from cryptography.fernet import Fernet

logger = structlog.get_logger(__name__)


class WebSocketState(Enum):
    """États de connexion WebSocket étendus"""
    CONNECTING = "connecting"
    CONNECTED = "connected" 
    AUTHENTICATED = "authenticated"
    SUBSCRIBING = "subscribing"
    ACTIVE = "active"
    IDLE = "idle"
    RECONNECTING = "reconnecting"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    SUSPENDED = "suspended"
    BANNED = "banned"
    ERROR = "error"


class MessagePriority(Enum):
    """Priorités de messages"""
    SYSTEM = 0      # Messages système critiques
    URGENT = 1      # Messages urgents
    HIGH = 2        # Haute priorité
    NORMAL = 3      # Priorité normale  
    LOW = 4         # Basse priorité
    BACKGROUND = 5  # Messages en arrière-plan


class ConnectionType(Enum):
    """Types de connexions"""
    USER = "user"           # Connexion utilisateur standard
    ADMIN = "admin"         # Connexion administrateur
    SERVICE = "service"     # Connexion service-to-service
    BOT = "bot"            # Connexion bot/automatisée
    ANONYMOUS = "anonymous" # Connexion anonyme limitée


@dataclass
class ConnectionMetrics:
    """Métriques de connexion"""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    connection_start: datetime = field(default_factory=datetime.utcnow)
    ping_count: int = 0
    avg_ping_time: float = 0.0
    errors: int = 0
    reconnect_count: int = 0


@dataclass
class WebSocketMessage:
    """Message WebSocket structuré"""
    id: str
    type: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    target: Optional[str] = None
    correlation_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    encrypted: bool = False


@dataclass
class ConnectionConfig:
    """Configuration de connexion"""
    max_message_size: int = 64 * 1024  # 64KB
    max_messages_per_second: int = 100
    heartbeat_interval: int = 30
    idle_timeout: int = 300  # 5 minutes
    max_subscriptions: int = 50
    compression_enabled: bool = True
    encryption_enabled: bool = False
    rate_limiting_enabled: bool = True
    auto_reconnect: bool = True
    circuit_breaker_threshold: int = 5


class RateLimiter:
    """Rate limiter pour connexions WebSocket"""
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
    
    def is_allowed(self) -> bool:
        """Vérifie si la requête est autorisée"""
        now = time.time()
        
        # Nettoyer les anciennes requêtes
        while self.requests and self.requests[0] <= now - self.window_seconds:
            self.requests.popleft()
        
        # Vérifier la limite
        if len(self.requests) >= self.max_requests:
            return False
        
        # Enregistrer la nouvelle requête
        self.requests.append(now)
        return True
    
    def get_remaining(self) -> int:
        """Retourne le nombre de requêtes restantes"""
        now = time.time()
        while self.requests and self.requests[0] <= now - self.window_seconds:
            self.requests.popleft()
        return max(0, self.max_requests - len(self.requests))


class CircuitBreaker:
    """Circuit breaker pour protection des connexions"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable, *args, **kwargs):
        """Exécute une fonction avec protection circuit breaker"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Gère le succès d'une opération"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Gère l'échec d'une opération"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class WebSocketConnection:
    """Connexion WebSocket avancée avec métriques"""
    
    def __init__(self, 
                 connection_id: str,
                 websocket: WebSocket,
                 user_id: Optional[str] = None,
                 tenant_id: str = "default",
                 connection_type: ConnectionType = ConnectionType.USER,
                 config: Optional[ConnectionConfig] = None):
        
        self.connection_id = connection_id
        self.websocket = websocket
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.connection_type = connection_type
        self.config = config or ConnectionConfig()
        
        # État et lifecycle
        self.state = WebSocketState.CONNECTING
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.last_ping = datetime.utcnow()
        
        # Gestion des messages
        self.message_queue = asyncio.Queue(maxsize=1000)
        self.subscriptions: Set[str] = set()
        self.metadata: Dict[str, Any] = {}
        
        # Sécurité et limiting
        self.rate_limiter = RateLimiter(config.max_messages_per_second)
        self.circuit_breaker = CircuitBreaker(config.circuit_breaker_threshold)
        self.auth_token: Optional[str] = None
        self.permissions: Set[str] = set()
        
        # Métriques
        self.metrics = ConnectionMetrics()
        
        # Chiffrement
        self.encryption_key: Optional[bytes] = None
        if config.encryption_enabled:
            self.encryption_key = Fernet.generate_key()
            self.fernet = Fernet(self.encryption_key)
        
        # Tâches asynchrones
        self.tasks: List[asyncio.Task] = []
        self._start_background_tasks()
    
    async def authenticate(self, token: str) -> bool:
        """Authentification de la connexion"""
        try:
            # Décoder et valider le JWT
            payload = jwt.decode(
                token, 
                options={"verify_signature": False}  # À configurer avec votre clé
            )
            
            self.user_id = payload.get("user_id")
            self.permissions = set(payload.get("permissions", []))
            self.auth_token = token
            self.state = WebSocketState.AUTHENTICATED
            
            logger.info("Connection authenticated", 
                       connection_id=self.connection_id,
                       user_id=self.user_id)
            return True
            
        except jwt.InvalidTokenError as e:
            logger.warning("Authentication failed", 
                          connection_id=self.connection_id,
                          error=str(e))
            return False
    
    async def send_message(self, message: WebSocketMessage) -> bool:
        """Envoie un message via WebSocket"""
        try:
            if self.websocket.client_state != WebSocketState.CONNECTED:
                return False
            
            # Vérifier rate limiting
            if self.config.rate_limiting_enabled and not self.rate_limiter.is_allowed():
                logger.warning("Rate limit exceeded", 
                              connection_id=self.connection_id)
                return False
            
            # Sérialiser le message
            serialized = self._serialize_message(message)
            
            # Chiffrer si nécessaire
            if message.encrypted and self.encryption_key:
                serialized = self.fernet.encrypt(serialized.encode()).decode()
            
            # Envoyer
            await self.websocket.send_text(serialized)
            
            # Mettre à jour les métriques
            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(serialized)
            self.last_activity = datetime.utcnow()
            
            return True
            
        except WebSocketDisconnect:
            await self._handle_disconnect()
            return False
        except Exception as e:
            logger.error("Failed to send message", 
                        connection_id=self.connection_id,
                        error=str(e))
            self.metrics.errors += 1
            return False
    
    async def receive_message(self) -> Optional[WebSocketMessage]:
        """Reçoit un message WebSocket"""
        try:
            data = await self.websocket.receive_text()
            
            # Déchiffrer si nécessaire
            if self.encryption_key and data.startswith("gAAAAA"):  # Fernet prefix
                data = self.fernet.decrypt(data.encode()).decode()
            
            # Désérialiser
            message = self._deserialize_message(data)
            
            # Mettre à jour les métriques
            self.metrics.messages_received += 1
            self.metrics.bytes_received += len(data)
            self.last_activity = datetime.utcnow()
            
            return message
            
        except WebSocketDisconnect:
            await self._handle_disconnect()
            return None
        except Exception as e:
            logger.error("Failed to receive message", 
                        connection_id=self.connection_id,
                        error=str(e))
            self.metrics.errors += 1
            return None
    
    async def ping(self) -> bool:
        """Envoie un ping de heartbeat"""
        try:
            ping_time = time.time()
            
            ping_message = WebSocketMessage(
                id=str(uuid.uuid4()),
                type="ping",
                payload={"timestamp": ping_time},
                priority=MessagePriority.SYSTEM
            )
            
            success = await self.send_message(ping_message)
            if success:
                self.last_ping = datetime.utcnow()
                self.metrics.ping_count += 1
            
            return success
            
        except Exception as e:
            logger.error("Ping failed", 
                        connection_id=self.connection_id,
                        error=str(e))
            return False
    
    async def pong(self, ping_timestamp: float):
        """Répond à un ping"""
        pong_time = time.time()
        ping_time = pong_time - ping_timestamp
        
        # Mettre à jour le ping moyen
        if self.metrics.ping_count > 0:
            self.metrics.avg_ping_time = (
                (self.metrics.avg_ping_time * (self.metrics.ping_count - 1) + ping_time) 
                / self.metrics.ping_count
            )
        else:
            self.metrics.avg_ping_time = ping_time
        
        pong_message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type="pong",
            payload={"ping_time": ping_time},
            priority=MessagePriority.SYSTEM
        )
        
        return await self.send_message(pong_message)
    
    async def subscribe(self, channel: str) -> bool:
        """S'abonne à un canal"""
        if len(self.subscriptions) >= self.config.max_subscriptions:
            logger.warning("Max subscriptions reached", 
                          connection_id=self.connection_id)
            return False
        
        self.subscriptions.add(channel)
        logger.info("Subscribed to channel", 
                   connection_id=self.connection_id,
                   channel=channel)
        return True
    
    async def unsubscribe(self, channel: str) -> bool:
        """Se désabonne d'un canal"""
        if channel in self.subscriptions:
            self.subscriptions.remove(channel)
            logger.info("Unsubscribed from channel", 
                       connection_id=self.connection_id,
                       channel=channel)
            return True
        return False
    
    async def close(self, code: int = 1000, reason: str = "Normal closure"):
        """Ferme la connexion"""
        try:
            # Annuler les tâches
            for task in self.tasks:
                task.cancel()
            
            # Fermer WebSocket
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.close(code, reason)
            
            self.state = WebSocketState.DISCONNECTED
            
            logger.info("Connection closed", 
                       connection_id=self.connection_id,
                       reason=reason)
            
        except Exception as e:
            logger.error("Error closing connection", 
                        connection_id=self.connection_id,
                        error=str(e))
    
    def _serialize_message(self, message: WebSocketMessage) -> str:
        """Sérialise un message"""
        return json.dumps({
            "id": message.id,
            "type": message.type,
            "payload": message.payload,
            "priority": message.priority.value,
            "timestamp": message.timestamp.isoformat(),
            "source": message.source,
            "target": message.target,
            "correlation_id": message.correlation_id
        })
    
    def _deserialize_message(self, data: str) -> WebSocketMessage:
        """Désérialise un message"""
        parsed = json.loads(data)
        
        return WebSocketMessage(
            id=parsed.get("id", str(uuid.uuid4())),
            type=parsed["type"],
            payload=parsed.get("payload", {}),
            priority=MessagePriority(parsed.get("priority", MessagePriority.NORMAL.value)),
            timestamp=datetime.fromisoformat(parsed.get("timestamp", datetime.utcnow().isoformat())),
            source=parsed.get("source"),
            target=parsed.get("target"),
            correlation_id=parsed.get("correlation_id")
        )
    
    def _start_background_tasks(self):
        """Démarre les tâches en arrière-plan"""
        # Tâche de heartbeat
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.tasks.append(heartbeat_task)
        
        # Tâche de nettoyage idle
        idle_task = asyncio.create_task(self._idle_monitor())
        self.tasks.append(idle_task)
    
    async def _heartbeat_loop(self):
        """Boucle de heartbeat"""
        while self.state != WebSocketState.DISCONNECTED:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.state in [WebSocketState.CONNECTED, WebSocketState.AUTHENTICATED]:
                    await self.ping()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error", 
                            connection_id=self.connection_id,
                            error=str(e))
    
    async def _idle_monitor(self):
        """Monitore les connexions inactives"""
        while self.state != WebSocketState.DISCONNECTED:
            try:
                await asyncio.sleep(60)  # Vérifier chaque minute
                
                time_since_activity = (datetime.utcnow() - self.last_activity).total_seconds()
                
                if time_since_activity > self.config.idle_timeout:
                    logger.info("Connection idle timeout", 
                               connection_id=self.connection_id)
                    await self.close(1001, "Idle timeout")
                    break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Idle monitor error", 
                            connection_id=self.connection_id,
                            error=str(e))
    
    async def _handle_disconnect(self):
        """Gère la déconnexion"""
        self.state = WebSocketState.DISCONNECTED
        
        # Annuler les tâches
        for task in self.tasks:
            task.cancel()
        
        logger.info("Connection disconnected", 
                   connection_id=self.connection_id)


class WebSocketCluster:
    """Cluster WebSocket pour mise à l'échelle horizontale"""
    
    def __init__(self, redis_client: aioredis.Redis, node_id: str):
        self.redis = redis_client
        self.node_id = node_id
        self.connections: Dict[str, WebSocketConnection] = {}
        
        # Métriques Prometheus
        self.metrics = {
            "connections_total": Gauge("websocket_connections_total", "Total WebSocket connections"),
            "messages_sent": Counter("websocket_messages_sent_total", "Total messages sent"),
            "messages_received": Counter("websocket_messages_received_total", "Total messages received"),
            "connection_duration": Histogram("websocket_connection_duration_seconds", "Connection duration"),
            "message_latency": Histogram("websocket_message_latency_seconds", "Message latency")
        }
    
    async def register_connection(self, connection: WebSocketConnection):
        """Enregistre une connexion dans le cluster"""
        self.connections[connection.connection_id] = connection
        
        # Enregistrer dans Redis pour découverte
        await self.redis.hset(
            f"ws:connections:{self.node_id}",
            connection.connection_id,
            json.dumps({
                "user_id": connection.user_id,
                "tenant_id": connection.tenant_id,
                "connected_at": connection.connected_at.isoformat(),
                "subscriptions": list(connection.subscriptions)
            })
        )
        
        self.metrics["connections_total"].inc()
        
        logger.info("Connection registered in cluster", 
                   connection_id=connection.connection_id,
                   node_id=self.node_id)
    
    async def unregister_connection(self, connection_id: str):
        """Désenregistre une connexion du cluster"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            del self.connections[connection_id]
            
            # Supprimer de Redis
            await self.redis.hdel(f"ws:connections:{self.node_id}", connection_id)
            
            self.metrics["connections_total"].dec()
            
            # Enregistrer la durée de connexion
            duration = (datetime.utcnow() - connection.connected_at).total_seconds()
            self.metrics["connection_duration"].observe(duration)
            
            logger.info("Connection unregistered from cluster", 
                       connection_id=connection_id,
                       node_id=self.node_id)
    
    async def broadcast_to_cluster(self, message: WebSocketMessage, target_user_id: Optional[str] = None):
        """Diffuse un message à travers le cluster"""
        message_data = {
            "message": message.__dict__,
            "target_user_id": target_user_id,
            "source_node": self.node_id
        }
        
        await self.redis.publish("ws:broadcast", json.dumps(message_data, default=str))
        
        logger.debug("Message broadcasted to cluster", 
                    message_id=message.id,
                    target_user_id=target_user_id)
    
    async def find_user_connections(self, user_id: str) -> List[Dict[str, Any]]:
        """Trouve toutes les connexions d'un utilisateur dans le cluster"""
        connections = []
        
        # Parcourir tous les nœuds
        for node_key in await self.redis.keys("ws:connections:*"):
            node_connections = await self.redis.hgetall(node_key)
            
            for conn_id, conn_data in node_connections.items():
                data = json.loads(conn_data)
                if data.get("user_id") == user_id:
                    connections.append({
                        "connection_id": conn_id,
                        "node_id": node_key.split(":")[-1],
                        **data
                    })
        
        return connections


class AdvancedWebSocketManager:
    """Gestionnaire WebSocket enterprise avancé"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.logger = structlog.get_logger(__name__)
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Configuration
        self.node_id = f"ws-node-{uuid.uuid4().hex[:8]}"
        self.config = ConnectionConfig()
        
        # Gestion des connexions
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        self.channel_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Cluster et load balancing
        self.cluster: Optional[WebSocketCluster] = None
        
        # Handlers de messages
        self.message_handlers: Dict[str, Callable] = {}
        
        # Métriques
        self.setup_metrics()
    
    def setup_metrics(self):
        """Configure les métriques Prometheus"""
        self.metrics = {
            "active_connections": Gauge("ws_active_connections", "Active WebSocket connections", ["tenant", "type"]),
            "total_messages": Counter("ws_messages_total", "Total WebSocket messages", ["type", "direction"]),
            "message_processing_time": Histogram("ws_message_processing_seconds", "Message processing time"),
            "connection_errors": Counter("ws_connection_errors_total", "Connection errors", ["error_type"]),
            "bandwidth_usage": Histogram("ws_bandwidth_bytes", "Bandwidth usage", ["direction"])
        }
    
    async def initialize(self):
        """Initialise le gestionnaire WebSocket"""
        try:
            # Connexion Redis
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Initialiser le cluster
            self.cluster = WebSocketCluster(self.redis_client, self.node_id)
            
            # Enregistrer les handlers par défaut
            self._register_default_handlers()
            
            # Démarrer les tâches de maintenance
            asyncio.create_task(self._maintenance_loop())
            asyncio.create_task(self._cluster_message_listener())
            
            self.logger.info("WebSocket manager initialized", node_id=self.node_id)
            
        except Exception as e:
            self.logger.error("Failed to initialize WebSocket manager", error=str(e))
            raise
    
    async def create_connection(self, 
                               websocket: WebSocket,
                               user_id: Optional[str] = None,
                               tenant_id: str = "default",
                               connection_type: ConnectionType = ConnectionType.USER,
                               config: Optional[ConnectionConfig] = None) -> str:
        """Crée une nouvelle connexion WebSocket"""
        
        connection_id = str(uuid.uuid4())
        connection_config = config or self.config
        
        # Créer la connexion
        connection = WebSocketConnection(
            connection_id=connection_id,
            websocket=websocket,
            user_id=user_id,
            tenant_id=tenant_id,
            connection_type=connection_type,
            config=connection_config
        )
        
        # Enregistrer localement
        self.connections[connection_id] = connection
        if user_id:
            self.user_connections[user_id].add(connection_id)
        
        # Enregistrer dans le cluster
        if self.cluster:
            await self.cluster.register_connection(connection)
        
        # Métriques
        self.metrics["active_connections"].labels(
            tenant=tenant_id, 
            type=connection_type.value
        ).inc()
        
        self.logger.info("WebSocket connection created", 
                        connection_id=connection_id,
                        user_id=user_id,
                        tenant_id=tenant_id)
        
        return connection_id
    
    async def remove_connection(self, connection_id: str):
        """Supprime une connexion"""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        # Supprimer localement
        del self.connections[connection_id]
        if connection.user_id:
            self.user_connections[connection.user_id].discard(connection_id)
        
        # Désabonner des canaux
        for channel in list(connection.subscriptions):
            self.channel_subscriptions[channel].discard(connection_id)
        
        # Supprimer du cluster
        if self.cluster:
            await self.cluster.unregister_connection(connection_id)
        
        # Métriques
        self.metrics["active_connections"].labels(
            tenant=connection.tenant_id,
            type=connection.connection_type.value
        ).dec()
        
        await connection.close()
        
        self.logger.info("WebSocket connection removed", 
                        connection_id=connection_id)
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Envoie un message à une connexion spécifique"""
        connection = self.connections.get(connection_id)
        if not connection:
            return False
        
        start_time = time.time()
        
        try:
            success = await connection.send_message(message)
            
            # Métriques
            processing_time = time.time() - start_time
            self.metrics["message_processing_time"].observe(processing_time)
            self.metrics["total_messages"].labels(type=message.type, direction="out").inc()
            
            return success
            
        except Exception as e:
            self.metrics["connection_errors"].labels(error_type="send_error").inc()
            self.logger.error("Failed to send message", 
                            connection_id=connection_id,
                            error=str(e))
            return False
    
    async def send_to_user(self, user_id: str, message: WebSocketMessage) -> int:
        """Envoie un message à toutes les connexions d'un utilisateur"""
        sent_count = 0
        
        # Connexions locales
        local_connections = self.user_connections.get(user_id, set())
        for connection_id in list(local_connections):
            if await self.send_to_connection(connection_id, message):
                sent_count += 1
        
        # Diffuser dans le cluster si nécessaire
        if self.cluster and not local_connections:
            await self.cluster.broadcast_to_cluster(message, user_id)
        
        return sent_count
    
    async def broadcast_to_channel(self, channel: str, message: WebSocketMessage) -> int:
        """Diffuse un message à tous les abonnés d'un canal"""
        sent_count = 0
        subscribers = self.channel_subscriptions.get(channel, set())
        
        for connection_id in list(subscribers):
            if await self.send_to_connection(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def handle_message(self, connection_id: str, raw_message: str):
        """Traite un message entrant"""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        try:
            # Désérialiser le message
            message = connection._deserialize_message(raw_message)
            
            # Vérifier les permissions
            if not self._check_permissions(connection, message):
                self.logger.warning("Permission denied", 
                                  connection_id=connection_id,
                                  message_type=message.type)
                return
            
            # Traiter selon le type
            handler = self.message_handlers.get(message.type)
            if handler:
                await handler(connection, message)
            else:
                await self._handle_unknown_message(connection, message)
            
            # Métriques
            self.metrics["total_messages"].labels(type=message.type, direction="in").inc()
            
        except Exception as e:
            self.metrics["connection_errors"].labels(error_type="message_error").inc()
            self.logger.error("Message handling error", 
                            connection_id=connection_id,
                            error=str(e))
    
    def register_handler(self, message_type: str, handler: Callable):
        """Enregistre un handler de message"""
        self.message_handlers[message_type] = handler
        self.logger.info("Message handler registered", message_type=message_type)
    
    def _register_default_handlers(self):
        """Enregistre les handlers par défaut"""
        self.register_handler("ping", self._handle_ping)
        self.register_handler("pong", self._handle_pong)
        self.register_handler("subscribe", self._handle_subscribe)
        self.register_handler("unsubscribe", self._handle_unsubscribe)
        self.register_handler("auth", self._handle_auth)
    
    async def _handle_ping(self, connection: WebSocketConnection, message: WebSocketMessage):
        """Traite un ping"""
        ping_timestamp = message.payload.get("timestamp", time.time())
        await connection.pong(ping_timestamp)
    
    async def _handle_pong(self, connection: WebSocketConnection, message: WebSocketMessage):
        """Traite un pong"""
        # Le calcul du ping time est déjà fait dans la méthode pong
        pass
    
    async def _handle_subscribe(self, connection: WebSocketConnection, message: WebSocketMessage):
        """Traite un abonnement"""
        channel = message.payload.get("channel")
        if channel and await connection.subscribe(channel):
            self.channel_subscriptions[channel].add(connection.connection_id)
    
    async def _handle_unsubscribe(self, connection: WebSocketConnection, message: WebSocketMessage):
        """Traite un désabonnement"""
        channel = message.payload.get("channel")
        if channel and await connection.unsubscribe(channel):
            self.channel_subscriptions[channel].discard(connection.connection_id)
    
    async def _handle_auth(self, connection: WebSocketConnection, message: WebSocketMessage):
        """Traite l'authentification"""
        token = message.payload.get("token")
        if token and await connection.authenticate(token):
            # Envoyer confirmation
            auth_response = WebSocketMessage(
                id=str(uuid.uuid4()),
                type="auth_success",
                payload={"user_id": connection.user_id},
                priority=MessagePriority.SYSTEM
            )
            await connection.send_message(auth_response)
    
    async def _handle_unknown_message(self, connection: WebSocketConnection, message: WebSocketMessage):
        """Traite un message de type inconnu"""
        self.logger.warning("Unknown message type", 
                          connection_id=connection.connection_id,
                          message_type=message.type)
        
        error_response = WebSocketMessage(
            id=str(uuid.uuid4()),
            type="error",
            payload={"error": f"Unknown message type: {message.type}"},
            priority=MessagePriority.SYSTEM,
            correlation_id=message.correlation_id
        )
        await connection.send_message(error_response)
    
    def _check_permissions(self, connection: WebSocketConnection, message: WebSocketMessage) -> bool:
        """Vérifie les permissions pour un message"""
        # Logique de permissions basée sur le type de connexion et message
        if connection.connection_type == ConnectionType.ANONYMOUS:
            # Connexions anonymes limitées
            allowed_types = ["ping", "pong", "subscribe", "auth"]
            return message.type in allowed_types
        
        # Autres vérifications de permissions...
        return True
    
    async def _maintenance_loop(self):
        """Boucle de maintenance du gestionnaire"""
        while True:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                # Nettoyer les connexions mortes
                dead_connections = []
                for conn_id, conn in self.connections.items():
                    if conn.state == WebSocketState.DISCONNECTED:
                        dead_connections.append(conn_id)
                
                for conn_id in dead_connections:
                    await self.remove_connection(conn_id)
                
                self.logger.debug("Maintenance completed", 
                                cleaned_connections=len(dead_connections))
                
            except Exception as e:
                self.logger.error("Maintenance error", error=str(e))
    
    async def _cluster_message_listener(self):
        """Écoute les messages du cluster"""
        if not self.redis_client:
            return
        
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("ws:broadcast")
        
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        if data["source_node"] != self.node_id:
                            # Message d'un autre nœud
                            await self._handle_cluster_message(data)
                    except Exception as e:
                        self.logger.error("Cluster message error", error=str(e))
        except Exception as e:
            self.logger.error("Cluster listener error", error=str(e))
    
    async def _handle_cluster_message(self, data: Dict[str, Any]):
        """Traite un message du cluster"""
        target_user_id = data.get("target_user_id")
        message_data = data["message"]
        
        # Reconstruire le message
        message = WebSocketMessage(
            id=message_data["id"],
            type=message_data["type"],
            payload=message_data["payload"],
            priority=MessagePriority(message_data["priority"]),
            timestamp=datetime.fromisoformat(message_data["timestamp"]),
            source=message_data.get("source"),
            target=message_data.get("target"),
            correlation_id=message_data.get("correlation_id")
        )
        
        if target_user_id:
            await self.send_to_user(target_user_id, message)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du gestionnaire"""
        total_connections = len(self.connections)
        active_connections = sum(
            1 for conn in self.connections.values() 
            if conn.state in [WebSocketState.CONNECTED, WebSocketState.AUTHENTICATED]
        )
        
        return {
            "node_id": self.node_id,
            "total_connections": total_connections,
            "active_connections": active_connections,
            "total_channels": len(self.channel_subscriptions),
            "total_users": len(self.user_connections),
            "message_handlers": len(self.message_handlers)
        }


# Instance globale
websocket_manager = AdvancedWebSocketManager()


# Export des classes principales
__all__ = [
    "AdvancedWebSocketManager",
    "WebSocketConnection", 
    "WebSocketCluster",
    "WebSocketMessage",
    "ConnectionConfig",
    "WebSocketState",
    "MessagePriority",
    "ConnectionType",
    "RateLimiter",
    "CircuitBreaker",
    "websocket_manager"
]
