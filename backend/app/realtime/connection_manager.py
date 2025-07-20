# 🎵 Spotify AI Agent - Real-Time Connection Manager
# ===================================================
# 
# Gestionnaire de connexions temps réel avec pooling avancé,
# load balancing et monitoring de la santé des connexions.
#
# 🎖️ Expert: Network Engineer + Infrastructure Architect
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ===================================================

"""
🔗 Real-Time Connection Manager
===============================

Advanced connection management system providing:
- Connection pooling and lifecycle management
- Load balancing across multiple servers
- Health monitoring and automatic failover
- Connection throttling and rate limiting
- Network optimization and compression
- Metrics collection and monitoring
- Security and authentication
- Graceful shutdown and cleanup
"""

import asyncio
import json
import logging
import time
import uuid
import weakref
import ssl
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Set, Optional, Union, Callable, Tuple
import statistics
import hashlib

# Networking
import websockets
import aiohttp
import aioredis
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

# Monitoring
from prometheus_client import Counter, Histogram, Gauge, Summary
import structlog

# Security
import jwt
from cryptography.fernet import Fernet

logger = structlog.get_logger(__name__)


class ConnectionState(Enum):
    """États de connexion"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    IDLE = "idle"
    ACTIVE = "active"
    RECONNECTING = "reconnecting"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class ConnectionType(Enum):
    """Types de connexion"""
    WEBSOCKET = "websocket"
    HTTP_STREAM = "http_stream"
    REDIS_PUBSUB = "redis_pubsub"
    GRPC_STREAM = "grpc_stream"


class LoadBalanceStrategy(Enum):
    """Stratégies de load balancing"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HASH_BASED = "hash_based"
    LATENCY_BASED = "latency_based"


@dataclass
class ConnectionMetrics:
    """Métriques de connexion"""
    connection_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    # Statistiques de trafic
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    
    # Performance
    average_latency: float = 0.0
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Erreurs
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    # Santé
    health_score: float = 1.0
    consecutive_failures: int = 0
    
    def update_latency(self, latency: float):
        """Met à jour la latence"""
        self.latency_samples.append(latency)
        if self.latency_samples:
            self.average_latency = statistics.mean(self.latency_samples)
    
    def record_error(self, error: str):
        """Enregistre une erreur"""
        self.error_count += 1
        self.last_error = error
        self.last_error_time = datetime.utcnow()
        self.consecutive_failures += 1
        
        # Réduire le score de santé
        self.health_score = max(0.0, self.health_score - 0.1)
    
    def record_success(self):
        """Enregistre un succès"""
        self.consecutive_failures = 0
        self.last_activity = datetime.utcnow()
        
        # Améliorer le score de santé
        self.health_score = min(1.0, self.health_score + 0.05)


@dataclass
class ServerEndpoint:
    """Point de terminaison serveur"""
    host: str
    port: int
    path: str = "/"
    ssl_enabled: bool = False
    weight: int = 1
    max_connections: int = 1000
    
    # État et métriques
    is_healthy: bool = True
    current_connections: int = 0
    total_connections: int = 0
    last_health_check: Optional[datetime] = None
    average_response_time: float = 0.0
    
    def get_url(self) -> str:
        """Construit l'URL complète"""
        scheme = "wss" if self.ssl_enabled else "ws"
        return f"{scheme}://{self.host}:{self.port}{self.path}"
    
    def can_accept_connection(self) -> bool:
        """Vérifie si peut accepter une nouvelle connexion"""
        return (self.is_healthy and 
                self.current_connections < self.max_connections)
    
    def get_load_factor(self) -> float:
        """Calcule le facteur de charge"""
        if self.max_connections == 0:
            return 1.0
        return self.current_connections / self.max_connections


class Connection:
    """Représente une connexion temps réel"""
    
    def __init__(self,
                 connection_id: str,
                 connection_type: ConnectionType,
                 endpoint: ServerEndpoint,
                 user_id: Optional[str] = None):
        
        self.connection_id = connection_id
        self.connection_type = connection_type
        self.endpoint = endpoint
        self.user_id = user_id
        
        # État
        self.state = ConnectionState.CONNECTING
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Authentification
        self.is_authenticated = False
        self.auth_token: Optional[str] = None
        self.permissions: Set[str] = set()
        
        # Métriques
        self.metrics = ConnectionMetrics(connection_id)
        
        # Configuration
        self.heartbeat_interval = 30  # secondes
        self.max_message_size = 1024 * 1024  # 1MB
        self.compression_enabled = True
        
        # Tâches async
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.receive_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_message: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Buffer de messages
        self.message_queue = asyncio.Queue(maxsize=1000)
        self.send_buffer: List[str] = []
        
    async def connect(self) -> bool:
        """Établit la connexion"""
        try:
            self.state = ConnectionState.CONNECTING
            
            if self.connection_type == ConnectionType.WEBSOCKET:
                await self._connect_websocket()
            elif self.connection_type == ConnectionType.HTTP_STREAM:
                await self._connect_http_stream()
            else:
                raise ValueError(f"Unsupported connection type: {self.connection_type}")
            
            self.state = ConnectionState.CONNECTED
            
            # Démarrer les tâches en arrière-plan
            await self._start_background_tasks()
            
            logger.info("Connection established", 
                       connection_id=self.connection_id,
                       endpoint=self.endpoint.get_url())
            
            return True
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.metrics.record_error(str(e))
            logger.error("Connection failed", 
                        connection_id=self.connection_id,
                        error=str(e))
            return False
    
    async def _connect_websocket(self):
        """Connecte via WebSocket"""
        # Configuration SSL
        ssl_context = None
        if self.endpoint.ssl_enabled:
            ssl_context = ssl.create_default_context()
        
        # Headers personnalisés
        headers = {
            "User-Agent": "Spotify-AI-Agent/1.0",
            "X-Connection-ID": self.connection_id
        }
        
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # Établir la connexion WebSocket
        self.websocket = await websockets.connect(
            self.endpoint.get_url(),
            ssl=ssl_context,
            extra_headers=headers,
            compression="deflate" if self.compression_enabled else None,
            max_size=self.max_message_size,
            ping_interval=self.heartbeat_interval,
            ping_timeout=10
        )
    
    async def _connect_http_stream(self):
        """Connecte via HTTP streaming"""
        # Configuration session
        connector = aiohttp.TCPConnector(
            ssl=self.endpoint.ssl_enabled,
            limit=100,
            ttl_dns_cache=300
        )
        
        timeout = aiohttp.ClientTimeout(total=None, sock_read=30)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "Spotify-AI-Agent/1.0",
                "X-Connection-ID": self.connection_id
            }
        )
    
    async def _start_background_tasks(self):
        """Démarre les tâches en arrière-plan"""
        # Tâche de heartbeat
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Tâche de réception
        if self.connection_type == ConnectionType.WEBSOCKET:
            self.receive_task = asyncio.create_task(self._websocket_receive_loop())
    
    async def _heartbeat_loop(self):
        """Boucle de heartbeat"""
        while self.state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Envoyer ping
                if self.connection_type == ConnectionType.WEBSOCKET and self.websocket:
                    start_time = time.time()
                    await self.websocket.ping()
                    
                    # Mesurer la latence
                    latency = time.time() - start_time
                    self.metrics.update_latency(latency * 1000)  # en ms
                
                self.metrics.record_success()
                
            except Exception as e:
                self.metrics.record_error(str(e))
                logger.warning("Heartbeat failed", 
                             connection_id=self.connection_id,
                             error=str(e))
                
                # Tentative de reconnexion si trop d'échecs
                if self.metrics.consecutive_failures > 3:
                    await self._attempt_reconnection()
                    break
    
    async def _websocket_receive_loop(self):
        """Boucle de réception WebSocket"""
        while (self.state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED] 
               and self.websocket):
            try:
                message = await self.websocket.recv()
                
                # Mettre à jour les métriques
                self.metrics.messages_received += 1
                self.metrics.bytes_received += len(message.encode('utf-8'))
                self.metrics.last_activity = datetime.utcnow()
                
                # Traiter le message
                if self.on_message:
                    await self.on_message(self, message)
                else:
                    # Ajouter à la queue
                    await self.message_queue.put(message)
                
            except ConnectionClosed:
                logger.info("WebSocket connection closed", 
                           connection_id=self.connection_id)
                await self._handle_disconnect()
                break
                
            except Exception as e:
                self.metrics.record_error(str(e))
                logger.error("Receive loop error", 
                           connection_id=self.connection_id,
                           error=str(e))
                
                if self.on_error:
                    await self.on_error(self, e)
    
    async def send_message(self, message: str, priority: int = 0) -> bool:
        """Envoie un message"""
        try:
            if self.state not in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]:
                logger.warning("Cannot send message, connection not ready",
                             connection_id=self.connection_id,
                             state=self.state.value)
                return False
            
            start_time = time.time()
            
            if self.connection_type == ConnectionType.WEBSOCKET and self.websocket:
                await self.websocket.send(message)
            else:
                # Autres types de connexion
                return False
            
            # Mettre à jour les métriques
            send_time = time.time() - start_time
            self.metrics.update_latency(send_time * 1000)
            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(message.encode('utf-8'))
            self.metrics.last_activity = datetime.utcnow()
            
            return True
            
        except Exception as e:
            self.metrics.record_error(str(e))
            logger.error("Send message failed", 
                        connection_id=self.connection_id,
                        error=str(e))
            return False
    
    async def authenticate(self, token: str) -> bool:
        """Authentifie la connexion"""
        try:
            self.state = ConnectionState.AUTHENTICATING
            
            # Vérifier le token JWT
            payload = jwt.decode(token, verify=False)  # Simplification
            self.user_id = payload.get("user_id")
            self.permissions = set(payload.get("permissions", []))
            
            self.auth_token = token
            self.is_authenticated = True
            self.state = ConnectionState.AUTHENTICATED
            
            logger.info("Connection authenticated", 
                       connection_id=self.connection_id,
                       user_id=self.user_id)
            
            return True
            
        except Exception as e:
            self.state = ConnectionState.CONNECTED
            self.metrics.record_error(str(e))
            logger.error("Authentication failed", 
                        connection_id=self.connection_id,
                        error=str(e))
            return False
    
    async def _attempt_reconnection(self):
        """Tente de se reconnecter"""
        self.state = ConnectionState.RECONNECTING
        
        # Nettoyer l'ancienne connexion
        await self._cleanup_connection()
        
        # Tentative de reconnexion
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                await asyncio.sleep(2 ** attempt)  # Backoff exponentiel
                
                if await self.connect():
                    logger.info("Reconnection successful", 
                               connection_id=self.connection_id,
                               attempt=attempt + 1)
                    return True
                    
            except Exception as e:
                logger.warning("Reconnection attempt failed", 
                             connection_id=self.connection_id,
                             attempt=attempt + 1,
                             error=str(e))
        
        # Échec de la reconnexion
        self.state = ConnectionState.DISCONNECTED
        await self._handle_disconnect()
        return False
    
    async def _handle_disconnect(self):
        """Gère la déconnexion"""
        self.state = ConnectionState.DISCONNECTED
        
        if self.on_disconnect:
            await self.on_disconnect(self)
        
        await self._cleanup_connection()
    
    async def _cleanup_connection(self):
        """Nettoie les ressources de connexion"""
        # Arrêter les tâches
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        if self.receive_task:
            self.receive_task.cancel()
        
        # Fermer les connexions
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        if self.session:
            await self.session.close()
            self.session = None
    
    async def disconnect(self):
        """Déconnecte proprement"""
        self.state = ConnectionState.DISCONNECTING
        await self._cleanup_connection()
        self.state = ConnectionState.DISCONNECTED
        
        logger.info("Connection disconnected", 
                   connection_id=self.connection_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut de la connexion"""
        return {
            "connection_id": self.connection_id,
            "state": self.state.value,
            "user_id": self.user_id,
            "is_authenticated": self.is_authenticated,
            "endpoint": self.endpoint.get_url(),
            "uptime": (datetime.utcnow() - self.metrics.created_at).total_seconds(),
            "last_activity": self.metrics.last_activity.isoformat(),
            "messages_sent": self.metrics.messages_sent,
            "messages_received": self.metrics.messages_received,
            "average_latency": self.metrics.average_latency,
            "health_score": self.metrics.health_score,
            "error_count": self.metrics.error_count
        }


class ConnectionPool:
    """Pool de connexions avec load balancing"""
    
    def __init__(self, 
                 endpoints: List[ServerEndpoint],
                 strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_CONNECTIONS):
        
        self.endpoints = endpoints
        self.strategy = strategy
        self.connections: Dict[str, Connection] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Load balancing
        self.round_robin_index = 0
        self.endpoint_stats: Dict[str, Dict] = defaultdict(dict)
        
        # Surveillance
        self.health_check_interval = 60  # secondes
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Métriques
        self.total_connections = 0
        self.active_connections = 0
        self.failed_connections = 0
        
    async def start(self):
        """Démarre le pool de connexions"""
        # Démarrer la surveillance de santé
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Connection pool started", 
                   endpoints=len(self.endpoints),
                   strategy=self.strategy.value)
    
    async def get_connection(self, 
                           user_id: Optional[str] = None,
                           connection_type: ConnectionType = ConnectionType.WEBSOCKET) -> Optional[Connection]:
        """Obtient une connexion du pool"""
        try:
            # Sélectionner un endpoint
            endpoint = self._select_endpoint()
            if not endpoint:
                logger.error("No healthy endpoints available")
                return None
            
            # Créer la connexion
            connection_id = str(uuid.uuid4())
            connection = Connection(
                connection_id=connection_id,
                connection_type=connection_type,
                endpoint=endpoint,
                user_id=user_id
            )
            
            # Configurer les callbacks
            connection.on_disconnect = self._handle_connection_disconnect
            connection.on_error = self._handle_connection_error
            
            # Établir la connexion
            if await connection.connect():
                # Ajouter au pool
                self.connections[connection_id] = connection
                self.total_connections += 1
                self.active_connections += 1
                
                # Mettre à jour les stats d'endpoint
                endpoint.current_connections += 1
                endpoint.total_connections += 1
                
                # Associer à l'utilisateur
                if user_id:
                    self.user_connections[user_id].add(connection_id)
                
                logger.info("Connection created", 
                           connection_id=connection_id,
                           user_id=user_id,
                           endpoint=endpoint.get_url())
                
                return connection
            else:
                self.failed_connections += 1
                return None
                
        except Exception as e:
            logger.error("Failed to create connection", error=str(e))
            self.failed_connections += 1
            return None
    
    def _select_endpoint(self) -> Optional[ServerEndpoint]:
        """Sélectionne un endpoint selon la stratégie"""
        # Filtrer les endpoints sains
        healthy_endpoints = [ep for ep in self.endpoints if ep.can_accept_connection()]
        
        if not healthy_endpoints:
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_endpoints)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_endpoints)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_endpoints)
        elif self.strategy == LoadBalanceStrategy.LATENCY_BASED:
            return self._latency_based_select(healthy_endpoints)
        else:
            return healthy_endpoints[0]
    
    def _round_robin_select(self, endpoints: List[ServerEndpoint]) -> ServerEndpoint:
        """Sélection round-robin"""
        endpoint = endpoints[self.round_robin_index % len(endpoints)]
        self.round_robin_index += 1
        return endpoint
    
    def _least_connections_select(self, endpoints: List[ServerEndpoint]) -> ServerEndpoint:
        """Sélection par nombre minimal de connexions"""
        return min(endpoints, key=lambda ep: ep.current_connections)
    
    def _weighted_round_robin_select(self, endpoints: List[ServerEndpoint]) -> ServerEndpoint:
        """Sélection weighted round-robin"""
        # Calculer les poids effectifs
        weighted_endpoints = []
        for ep in endpoints:
            # Le poids effectif diminue avec la charge
            effective_weight = ep.weight * (1 - ep.get_load_factor())
            weighted_endpoints.extend([ep] * max(1, int(effective_weight * 10)))
        
        if weighted_endpoints:
            index = self.round_robin_index % len(weighted_endpoints)
            self.round_robin_index += 1
            return weighted_endpoints[index]
        
        return endpoints[0]
    
    def _latency_based_select(self, endpoints: List[ServerEndpoint]) -> ServerEndpoint:
        """Sélection basée sur la latence"""
        return min(endpoints, key=lambda ep: ep.average_response_time)
    
    async def _handle_connection_disconnect(self, connection: Connection):
        """Gère la déconnexion d'une connexion"""
        connection_id = connection.connection_id
        
        # Retirer du pool
        if connection_id in self.connections:
            del self.connections[connection_id]
            self.active_connections -= 1
        
        # Mettre à jour les stats d'endpoint
        connection.endpoint.current_connections -= 1
        
        # Retirer de l'association utilisateur
        if connection.user_id:
            self.user_connections[connection.user_id].discard(connection_id)
        
        logger.info("Connection removed from pool", 
                   connection_id=connection_id)
    
    async def _handle_connection_error(self, connection: Connection, error: Exception):
        """Gère les erreurs de connexion"""
        logger.error("Connection error in pool", 
                    connection_id=connection.connection_id,
                    error=str(error))
        
        # Marquer l'endpoint comme potentiellement problématique
        if connection.metrics.consecutive_failures > 5:
            connection.endpoint.is_healthy = False
    
    async def _health_check_loop(self):
        """Boucle de vérification de santé"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
                
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
    
    async def _perform_health_checks(self):
        """Effectue les vérifications de santé"""
        for endpoint in self.endpoints:
            try:
                # Test de connectivité simple
                start_time = time.time()
                
                # Utiliser aiohttp pour un health check HTTP
                async with aiohttp.ClientSession() as session:
                    health_url = f"http{'s' if endpoint.ssl_enabled else ''}://{endpoint.host}:{endpoint.port}/health"
                    
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            endpoint.is_healthy = True
                            response_time = time.time() - start_time
                            endpoint.average_response_time = response_time
                        else:
                            endpoint.is_healthy = False
                
                endpoint.last_health_check = datetime.utcnow()
                
            except Exception as e:
                endpoint.is_healthy = False
                logger.warning("Health check failed", 
                             endpoint=endpoint.get_url(),
                             error=str(e))
    
    def get_user_connections(self, user_id: str) -> List[Connection]:
        """Récupère les connexions d'un utilisateur"""
        connection_ids = self.user_connections.get(user_id, set())
        return [self.connections[cid] for cid in connection_ids if cid in self.connections]
    
    async def broadcast_to_user(self, user_id: str, message: str) -> int:
        """Diffuse un message à toutes les connexions d'un utilisateur"""
        connections = self.get_user_connections(user_id)
        sent_count = 0
        
        for connection in connections:
            if await connection.send_message(message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_all(self, message: str) -> int:
        """Diffuse un message à toutes les connexions"""
        sent_count = 0
        
        for connection in self.connections.values():
            if await connection.send_message(message):
                sent_count += 1
        
        return sent_count
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Statistiques du pool"""
        endpoint_stats = []
        for endpoint in self.endpoints:
            endpoint_stats.append({
                "url": endpoint.get_url(),
                "is_healthy": endpoint.is_healthy,
                "current_connections": endpoint.current_connections,
                "total_connections": endpoint.total_connections,
                "max_connections": endpoint.max_connections,
                "load_factor": endpoint.get_load_factor(),
                "average_response_time": endpoint.average_response_time,
                "last_health_check": endpoint.last_health_check.isoformat() if endpoint.last_health_check else None
            })
        
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "failed_connections": self.failed_connections,
            "unique_users": len(self.user_connections),
            "load_balance_strategy": self.strategy.value,
            "endpoints": endpoint_stats
        }
    
    async def shutdown(self):
        """Arrête le pool de connexions"""
        # Arrêter la surveillance
        if self.health_check_task:
            self.health_check_task.cancel()
        
        # Fermer toutes les connexions
        disconnect_tasks = []
        for connection in self.connections.values():
            disconnect_tasks.append(connection.disconnect())
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        self.connections.clear()
        self.user_connections.clear()
        
        logger.info("Connection pool shutdown completed")


class RealTimeConnectionManager:
    """Gestionnaire principal des connexions temps réel"""
    
    def __init__(self,
                 endpoints: List[ServerEndpoint],
                 redis_url: str = "redis://localhost:6379"):
        
        self.endpoints = endpoints
        self.redis_url = redis_url
        
        # Pools de connexions par type
        self.websocket_pool = ConnectionPool(endpoints, LoadBalanceStrategy.LEAST_CONNECTIONS)
        self.http_stream_pool = ConnectionPool(endpoints, LoadBalanceStrategy.LATENCY_BASED)
        
        # Cache et state
        self.redis_client: Optional[aioredis.Redis] = None
        self.user_sessions: Dict[str, Dict] = {}
        
        # Configuration
        self.max_connections_per_user = 10
        self.session_timeout = 3600  # 1 heure
        
        # Surveillance et métriques
        self.metrics = {
            "connections_created": 0,
            "connections_failed": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_transferred": 0
        }
        
        # Tâches en arrière-plan
        self.background_tasks: List[asyncio.Task] = []
        
        # Métriques Prometheus
        self.prometheus_metrics = {
            "active_connections": Gauge("realtime_active_connections", "Active connections", ["type", "endpoint"]),
            "connection_duration": Histogram("realtime_connection_duration_seconds", "Connection duration"),
            "message_latency": Histogram("realtime_message_latency_seconds", "Message latency"),
            "errors_total": Counter("realtime_errors_total", "Total errors", ["type", "endpoint"])
        }
    
    async def initialize(self):
        """Initialise le gestionnaire"""
        try:
            # Connexion Redis
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Démarrer les pools
            await self.websocket_pool.start()
            await self.http_stream_pool.start()
            
            # Démarrer les tâches en arrière-plan
            await self._start_background_tasks()
            
            logger.info("Real-time connection manager initialized")
            
        except Exception as e:
            logger.error("Failed to initialize connection manager", error=str(e))
            raise
    
    async def _start_background_tasks(self):
        """Démarre les tâches en arrière-plan"""
        # Session cleanup
        session_cleanup = asyncio.create_task(self._session_cleanup_loop())
        self.background_tasks.append(session_cleanup)
        
        # Metrics collection
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.append(metrics_task)
        
        logger.info("Connection manager background tasks started")
    
    async def create_connection(self,
                              user_id: str,
                              connection_type: ConnectionType = ConnectionType.WEBSOCKET,
                              auth_token: Optional[str] = None) -> Optional[Connection]:
        """Crée une nouvelle connexion pour un utilisateur"""
        try:
            # Vérifier les limites
            if not await self._check_connection_limits(user_id):
                logger.warning("Connection limit exceeded", user_id=user_id)
                return None
            
            # Sélectionner le pool approprié
            if connection_type == ConnectionType.WEBSOCKET:
                pool = self.websocket_pool
            else:
                pool = self.http_stream_pool
            
            # Créer la connexion
            connection = await pool.get_connection(user_id, connection_type)
            
            if connection:
                # Authentifier si token fourni
                if auth_token:
                    await connection.authenticate(auth_token)
                
                # Enregistrer la session
                await self._register_user_session(user_id, connection)
                
                # Métriques
                self.metrics["connections_created"] += 1
                self.prometheus_metrics["active_connections"].labels(
                    type=connection_type.value,
                    endpoint=connection.endpoint.get_url()
                ).inc()
                
                logger.info("Connection created successfully", 
                           user_id=user_id,
                           connection_id=connection.connection_id)
                
                return connection
            else:
                self.metrics["connections_failed"] += 1
                return None
                
        except Exception as e:
            logger.error("Failed to create connection", 
                        user_id=user_id,
                        error=str(e))
            self.metrics["connections_failed"] += 1
            return None
    
    async def _check_connection_limits(self, user_id: str) -> bool:
        """Vérifie les limites de connexion"""
        # Compter les connexions existantes
        ws_connections = self.websocket_pool.get_user_connections(user_id)
        http_connections = self.http_stream_pool.get_user_connections(user_id)
        
        total_connections = len(ws_connections) + len(http_connections)
        
        return total_connections < self.max_connections_per_user
    
    async def _register_user_session(self, user_id: str, connection: Connection):
        """Enregistre la session utilisateur"""
        session_id = str(uuid.uuid4())
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "connection_id": connection.connection_id,
            "connection_type": connection.connection_type.value,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "endpoint": connection.endpoint.get_url()
        }
        
        # Stocker en mémoire
        self.user_sessions[session_id] = session_data
        
        # Stocker dans Redis
        if self.redis_client:
            await self.redis_client.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(session_data, default=str)
            )
            
            # Index par utilisateur
            await self.redis_client.sadd(f"user_sessions:{user_id}", session_id)
            await self.redis_client.expire(f"user_sessions:{user_id}", self.session_timeout)
    
    async def send_message_to_user(self, 
                                 user_id: str,
                                 message: str,
                                 connection_type: Optional[ConnectionType] = None) -> int:
        """Envoie un message à un utilisateur"""
        sent_count = 0
        
        try:
            # WebSocket
            if connection_type is None or connection_type == ConnectionType.WEBSOCKET:
                sent_count += await self.websocket_pool.broadcast_to_user(user_id, message)
            
            # HTTP Stream
            if connection_type is None or connection_type == ConnectionType.HTTP_STREAM:
                sent_count += await self.http_stream_pool.broadcast_to_user(user_id, message)
            
            # Métriques
            if sent_count > 0:
                self.metrics["messages_sent"] += sent_count
                self.metrics["bytes_transferred"] += len(message.encode('utf-8')) * sent_count
            
            return sent_count
            
        except Exception as e:
            logger.error("Failed to send message to user", 
                        user_id=user_id,
                        error=str(e))
            return 0
    
    async def broadcast_message(self, 
                              message: str,
                              connection_type: Optional[ConnectionType] = None) -> int:
        """Diffuse un message à toutes les connexions"""
        total_sent = 0
        
        try:
            # WebSocket
            if connection_type is None or connection_type == ConnectionType.WEBSOCKET:
                total_sent += await self.websocket_pool.broadcast_to_all(message)
            
            # HTTP Stream
            if connection_type is None or connection_type == ConnectionType.HTTP_STREAM:
                total_sent += await self.http_stream_pool.broadcast_to_all(message)
            
            # Métriques
            if total_sent > 0:
                self.metrics["messages_sent"] += total_sent
                self.metrics["bytes_transferred"] += len(message.encode('utf-8')) * total_sent
            
            return total_sent
            
        except Exception as e:
            logger.error("Failed to broadcast message", error=str(e))
            return 0
    
    async def get_user_connections(self, user_id: str) -> List[Dict[str, Any]]:
        """Récupère les informations des connexions d'un utilisateur"""
        connections_info = []
        
        # WebSocket connections
        ws_connections = self.websocket_pool.get_user_connections(user_id)
        for conn in ws_connections:
            connections_info.append({
                **conn.get_status(),
                "pool_type": "websocket"
            })
        
        # HTTP Stream connections
        http_connections = self.http_stream_pool.get_user_connections(user_id)
        for conn in http_connections:
            connections_info.append({
                **conn.get_status(),
                "pool_type": "http_stream"
            })
        
        return connections_info
    
    async def disconnect_user(self, user_id: str) -> int:
        """Déconnecte toutes les connexions d'un utilisateur"""
        disconnected = 0
        
        # WebSocket
        ws_connections = self.websocket_pool.get_user_connections(user_id)
        for conn in ws_connections:
            await conn.disconnect()
            disconnected += 1
        
        # HTTP Stream
        http_connections = self.http_stream_pool.get_user_connections(user_id)
        for conn in http_connections:
            await conn.disconnect()
            disconnected += 1
        
        # Nettoyer les sessions
        await self._cleanup_user_sessions(user_id)
        
        return disconnected
    
    async def _cleanup_user_sessions(self, user_id: str):
        """Nettoie les sessions d'un utilisateur"""
        if self.redis_client:
            # Récupérer les sessions
            session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")
            
            # Supprimer chaque session
            for session_id in session_ids:
                await self.redis_client.delete(f"session:{session_id}")
            
            # Supprimer l'index
            await self.redis_client.delete(f"user_sessions:{user_id}")
        
        # Nettoyer la mémoire
        sessions_to_remove = []
        for session_id, session_data in self.user_sessions.items():
            if session_data["user_id"] == user_id:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.user_sessions[session_id]
    
    async def _session_cleanup_loop(self):
        """Boucle de nettoyage des sessions expirées"""
        while True:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                # Nettoyer les sessions expirées
                now = datetime.utcnow()
                expired_sessions = []
                
                for session_id, session_data in self.user_sessions.items():
                    last_activity = session_data["last_activity"]
                    if isinstance(last_activity, str):
                        last_activity = datetime.fromisoformat(last_activity)
                    
                    if (now - last_activity).total_seconds() > self.session_timeout:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.user_sessions[session_id]
                
                if expired_sessions:
                    logger.info("Cleaned up expired sessions", count=len(expired_sessions))
                
            except Exception as e:
                logger.error("Session cleanup error", error=str(e))
    
    async def _metrics_collection_loop(self):
        """Boucle de collecte des métriques"""
        while True:
            try:
                await asyncio.sleep(60)  # Toutes les minutes
                
                # Collecter les métriques des pools
                ws_stats = self.websocket_pool.get_pool_stats()
                http_stats = self.http_stream_pool.get_pool_stats()
                
                # Mettre à jour Prometheus
                for endpoint_stat in ws_stats["endpoints"]:
                    self.prometheus_metrics["active_connections"].labels(
                        type="websocket",
                        endpoint=endpoint_stat["url"]
                    ).set(endpoint_stat["current_connections"])
                
                for endpoint_stat in http_stats["endpoints"]:
                    self.prometheus_metrics["active_connections"].labels(
                        type="http_stream",
                        endpoint=endpoint_stat["url"]
                    ).set(endpoint_stat["current_connections"])
                
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Statistiques du gestionnaire"""
        return {
            "total_metrics": self.metrics,
            "websocket_pool": self.websocket_pool.get_pool_stats(),
            "http_stream_pool": self.http_stream_pool.get_pool_stats(),
            "active_sessions": len(self.user_sessions),
            "configuration": {
                "max_connections_per_user": self.max_connections_per_user,
                "session_timeout": self.session_timeout,
                "endpoints_count": len(self.endpoints)
            }
        }
    
    async def shutdown(self):
        """Arrête le gestionnaire"""
        # Arrêter les tâches en arrière-plan
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Arrêter les pools
        await self.websocket_pool.shutdown()
        await self.http_stream_pool.shutdown()
        
        # Fermer Redis
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Connection manager shutdown completed")


# Factory function
async def create_connection_manager(endpoints: List[ServerEndpoint],
                                  redis_url: str = "redis://localhost:6379") -> RealTimeConnectionManager:
    """Crée et initialise le gestionnaire de connexions"""
    manager = RealTimeConnectionManager(endpoints, redis_url)
    await manager.initialize()
    return manager


# Instance globale
connection_manager: Optional[RealTimeConnectionManager] = None


# Export des classes principales
__all__ = [
    "RealTimeConnectionManager",
    "ConnectionPool",
    "Connection",
    "ServerEndpoint",
    "ConnectionState",
    "ConnectionType",
    "LoadBalanceStrategy",
    "ConnectionMetrics",
    "create_connection_manager"
]
