"""
🔧 Connection Manager - Gestionnaire de Connexions Ultra-Avancé
=============================================================

Gestionnaire de connexions pour l'isolation des données multi-tenant
avec pooling intelligent, load balancing, et monitoring avancé.

Author: Architecte Base de Données - Fahed Mlaiel
"""

import asyncio
import logging
import weakref
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from contextlib import asynccontextmanager
import hashlib
import json

from ..core.tenant_context import TenantContext, TenantType
from ..exceptions import DataIsolationError, ConnectionError


class ConnectionType(Enum):
    """Types de connexion"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    ANALYTICS = "analytics"
    BACKUP = "backup"


class ConnectionState(Enum):
    """États de connexion"""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    CLOSED = "closed"
    MAINTENANCE = "maintenance"


class LoadBalancingStrategy(Enum):
    """Stratégies de load balancing"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    RESOURCE_BASED = "resource_based"


@dataclass
class ConnectionMetrics:
    """Métriques d'une connexion"""
    connection_id: str
    created_at: datetime
    last_used: datetime
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_response_time: float = 0.0
    current_connections: int = 0
    max_connections_used: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class DatabaseConnectionConfig:
    """Configuration de connexion à une base de données"""
    host: str
    port: int
    database: str
    username: str
    password: str
    
    # Pool settings
    min_size: int = 5
    max_size: int = 20
    max_queries: int = 50000
    max_inactive_connection_lifetime: float = 300.0
    
    # Connection settings
    connection_timeout: float = 10.0
    command_timeout: Optional[float] = None
    ssl_enabled: bool = False
    ssl_context: Optional[Any] = None
    
    # Advanced settings
    statement_cache_size: int = 1024
    prepared_statement_cache_size: int = 256
    connection_class: Optional[str] = None
    
    # Tenant-specific settings
    tenant_id: Optional[str] = None
    tenant_type: Optional[TenantType] = None
    connection_type: ConnectionType = ConnectionType.READ
    
    # Weights for load balancing
    weight: int = 100
    priority: int = 1
    
    # Health check settings
    health_check_interval: int = 30
    health_check_query: str = "SELECT 1"
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolConfiguration:
    """Configuration globale du pool"""
    # Pool sizing
    default_min_size: int = 5
    default_max_size: int = 50
    max_total_connections: int = 1000
    
    # Connection lifecycle
    connection_timeout: float = 10.0
    max_connection_age: int = 3600
    idle_timeout: int = 300
    
    # Load balancing
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS
    health_check_interval: int = 30
    
    # Circuit breaker
    enable_circuit_breaker: bool = True
    failure_threshold: int = 5
    recovery_timeout: int = 60
    
    # Monitoring
    enable_metrics: bool = True
    metrics_retention_days: int = 7
    slow_query_threshold: float = 1.0
    
    # Security
    encrypt_connections: bool = True
    validate_certificates: bool = True
    
    # Failover
    enable_automatic_failover: bool = True
    failover_timeout: int = 30
    max_failover_attempts: int = 3


class ConnectionPool:
    """Pool de connexions pour un tenant/database spécifique"""
    
    def __init__(
        self, 
        pool_id: str,
        config: DatabaseConnectionConfig,
        pool_config: PoolConfiguration
    ):
        self.pool_id = pool_id
        self.config = config
        self.pool_config = pool_config
        self.logger = logging.getLogger(f"connection_pool.{pool_id}")
        
        # Connection tracking
        self._connections: Dict[str, Any] = {}  # connection_id -> connection
        self._connection_states: Dict[str, ConnectionState] = {}
        self._connection_metrics: Dict[str, ConnectionMetrics] = {}
        
        # Pool state
        self._is_initialized = False
        self._is_healthy = True
        self._total_connections = 0
        self._active_connections = 0
        self._creation_lock = asyncio.Lock()
        
        # Circuit breaker state
        self._circuit_breaker_state = "closed"  # closed, open, half_open
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        
        # Load balancing
        self._round_robin_counter = 0
        self._connection_weights: Dict[str, int] = {}
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_health_check: Optional[datetime] = None
        
        # Performance tracking
        self._query_history: List[Dict[str, Any]] = []
        self._performance_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0.0,
            "min_response_time": float('inf'),
            "max_response_time": 0.0
        }
    
    async def initialize(self):
        """Initialise le pool de connexions"""
        try:
            async with self._creation_lock:
                if self._is_initialized:
                    return
                
                # Create initial connections
                for i in range(self.config.min_size):
                    await self._create_connection()
                
                # Start health check task
                if self.pool_config.health_check_interval > 0:
                    self._health_check_task = asyncio.create_task(
                        self._health_check_loop()
                    )
                
                self._is_initialized = True
                self.logger.info(f"Pool {self.pool_id} initialized with {len(self._connections)} connections")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize pool {self.pool_id}: {e}")
            raise ConnectionError(f"Pool initialization failed: {e}")
    
    async def _create_connection(self) -> str:
        """Crée une nouvelle connexion"""
        if self._total_connections >= self.config.max_size:
            raise ConnectionError(f"Pool {self.pool_id} reached max size ({self.config.max_size})")
        
        connection_id = f"{self.pool_id}_{self._total_connections}_{int(datetime.now().timestamp())}"
        
        try:
            # Create database connection (mock implementation)
            # In real implementation, this would use asyncpg, aiomysql, etc.
            connection = await self._create_database_connection()
            
            # Track connection
            self._connections[connection_id] = connection
            self._connection_states[connection_id] = ConnectionState.IDLE
            self._connection_metrics[connection_id] = ConnectionMetrics(
                connection_id=connection_id,
                created_at=datetime.now(timezone.utc),
                last_used=datetime.now(timezone.utc)
            )
            self._connection_weights[connection_id] = self.config.weight
            
            self._total_connections += 1
            
            self.logger.debug(f"Created connection {connection_id}")
            return connection_id
            
        except Exception as e:
            self.logger.error(f"Failed to create connection: {e}")
            raise ConnectionError(f"Connection creation failed: {e}")
    
    async def _create_database_connection(self) -> Any:
        """Crée une connexion database réelle"""
        # Mock implementation - replace with actual database connection
        await asyncio.sleep(0.1)  # Simulate connection time
        return {
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database,
            "connected_at": datetime.now(timezone.utc),
            "mock": True
        }
    
    @asynccontextmanager
    async def acquire_connection(self, connection_type: ConnectionType = ConnectionType.READ):
        """Acquiert une connexion du pool"""
        connection_id = None
        try:
            # Check circuit breaker
            if not self._circuit_breaker_check():
                raise ConnectionError("Circuit breaker is open")
            
            # Get available connection
            connection_id = await self._get_available_connection(connection_type)
            if not connection_id:
                # Try to create new connection if pool not full
                if self._total_connections < self.config.max_size:
                    connection_id = await self._create_connection()
                else:
                    raise ConnectionError("No available connections and pool is full")
            
            # Mark connection as busy
            self._connection_states[connection_id] = ConnectionState.BUSY
            self._active_connections += 1
            
            # Update metrics
            metrics = self._connection_metrics[connection_id]
            metrics.last_used = datetime.now(timezone.utc)
            metrics.current_connections += 1
            metrics.max_connections_used = max(
                metrics.max_connections_used, 
                metrics.current_connections
            )
            
            connection = self._connections[connection_id]
            
            yield connection
            
        except Exception as e:
            self._record_failure(connection_id, str(e))
            raise
        finally:
            if connection_id:
                # Release connection
                self._connection_states[connection_id] = ConnectionState.IDLE
                self._active_connections -= 1
                
                # Update metrics
                if connection_id in self._connection_metrics:
                    self._connection_metrics[connection_id].current_connections -= 1
    
    async def _get_available_connection(self, connection_type: ConnectionType) -> Optional[str]:
        """Obtient une connexion disponible selon la stratégie de load balancing"""
        available_connections = [
            conn_id for conn_id, state in self._connection_states.items()
            if state == ConnectionState.IDLE
        ]
        
        if not available_connections:
            return None
        
        strategy = self.pool_config.load_balancing_strategy
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_connections)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_connections)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_connections)
        elif strategy == LoadBalancingStrategy.RESPONSE_TIME:
            return self._response_time_selection(available_connections)
        elif strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_selection(available_connections)
        else:
            return available_connections[0]  # Fallback
    
    def _round_robin_selection(self, connections: List[str]) -> str:
        """Sélection round-robin"""
        if not connections:
            return None
        
        selected = connections[self._round_robin_counter % len(connections)]
        self._round_robin_counter += 1
        return selected
    
    def _least_connections_selection(self, connections: List[str]) -> str:
        """Sélection basée sur le moins de connexions actives"""
        return min(
            connections,
            key=lambda conn_id: self._connection_metrics[conn_id].current_connections
        )
    
    def _weighted_round_robin_selection(self, connections: List[str]) -> str:
        """Sélection weighted round-robin"""
        # Implement weighted selection based on connection weights
        total_weight = sum(self._connection_weights.get(conn_id, 100) for conn_id in connections)
        
        if total_weight == 0:
            return connections[0]
        
        # Simple weighted selection
        import random
        threshold = random.randint(1, total_weight)
        current_weight = 0
        
        for conn_id in connections:
            current_weight += self._connection_weights.get(conn_id, 100)
            if current_weight >= threshold:
                return conn_id
        
        return connections[0]  # Fallback
    
    def _response_time_selection(self, connections: List[str]) -> str:
        """Sélection basée sur le temps de réponse"""
        return min(
            connections,
            key=lambda conn_id: self._connection_metrics[conn_id].avg_response_time
        )
    
    def _resource_based_selection(self, connections: List[str]) -> str:
        """Sélection basée sur les ressources"""
        # Select connection with best resource utilization
        return min(
            connections,
            key=lambda conn_id: (
                self._connection_metrics[conn_id].current_connections * 0.4 +
                self._connection_metrics[conn_id].avg_response_time * 0.6
            )
        )
    
    def _circuit_breaker_check(self) -> bool:
        """Vérifie l'état du circuit breaker"""
        if not self.pool_config.enable_circuit_breaker:
            return True
        
        now = datetime.now(timezone.utc)
        
        if self._circuit_breaker_state == "open":
            # Check if recovery timeout has passed
            if (self._last_failure_time and 
                (now - self._last_failure_time).total_seconds() > self.pool_config.recovery_timeout):
                self._circuit_breaker_state = "half_open"
                self.logger.info(f"Circuit breaker for pool {self.pool_id} is now half-open")
                return True
            return False
        
        return True  # closed or half_open
    
    def _record_failure(self, connection_id: Optional[str], error: str):
        """Enregistre un échec"""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)
        
        if connection_id and connection_id in self._connection_metrics:
            metrics = self._connection_metrics[connection_id]
            metrics.failed_queries += 1
            metrics.error_count += 1
            metrics.last_error = error
        
        # Update performance stats
        self._performance_stats["failed_queries"] += 1
        
        # Check circuit breaker
        if (self.pool_config.enable_circuit_breaker and 
            self._failure_count >= self.pool_config.failure_threshold):
            self._circuit_breaker_state = "open"
            self.logger.warning(f"Circuit breaker opened for pool {self.pool_id}")
    
    def _record_success(self, connection_id: str, response_time: float):
        """Enregistre un succès"""
        if self._circuit_breaker_state == "half_open":
            self._circuit_breaker_state = "closed"
            self._failure_count = 0
            self.logger.info(f"Circuit breaker closed for pool {self.pool_id}")
        
        # Update connection metrics
        if connection_id in self._connection_metrics:
            metrics = self._connection_metrics[connection_id]
            metrics.successful_queries += 1
            metrics.total_queries += 1
            
            # Update average response time (exponential moving average)
            if metrics.avg_response_time == 0.0:
                metrics.avg_response_time = response_time
            else:
                metrics.avg_response_time = (
                    metrics.avg_response_time * 0.9 + response_time * 0.1
                )
        
        # Update performance stats
        self._performance_stats["successful_queries"] += 1
        self._performance_stats["total_queries"] += 1
        
        # Update min/max response times
        self._performance_stats["min_response_time"] = min(
            self._performance_stats["min_response_time"], response_time
        )
        self._performance_stats["max_response_time"] = max(
            self._performance_stats["max_response_time"], response_time
        )
        
        # Update average response time
        total = self._performance_stats["total_queries"]
        if total > 0:
            current_avg = self._performance_stats["avg_response_time"]
            self._performance_stats["avg_response_time"] = (
                current_avg * (total - 1) + response_time
            ) / total
    
    async def _health_check_loop(self):
        """Boucle de vérification de santé"""
        while True:
            try:
                await asyncio.sleep(self.pool_config.health_check_interval)
                await self._perform_health_check()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error for pool {self.pool_id}: {e}")
    
    async def _perform_health_check(self):
        """Effectue une vérification de santé"""
        self._last_health_check = datetime.now(timezone.utc)
        healthy_connections = 0
        
        for connection_id, connection in self._connections.items():
            try:
                # Perform health check query
                start_time = datetime.now(timezone.utc)
                await self._execute_health_check(connection)
                response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                # Mark as healthy
                if self._connection_states[connection_id] == ConnectionState.ERROR:
                    self._connection_states[connection_id] = ConnectionState.IDLE
                    self.logger.info(f"Connection {connection_id} recovered")
                
                self._record_success(connection_id, response_time)
                healthy_connections += 1
                
            except Exception as e:
                # Mark as unhealthy
                self._connection_states[connection_id] = ConnectionState.ERROR
                self._record_failure(connection_id, str(e))
                self.logger.warning(f"Connection {connection_id} health check failed: {e}")
        
        # Update pool health status
        self._is_healthy = healthy_connections > 0
        
        if not self._is_healthy:
            self.logger.error(f"Pool {self.pool_id} is unhealthy - no healthy connections")
    
    async def _execute_health_check(self, connection: Any):
        """Exécute une requête de health check"""
        # Mock implementation
        await asyncio.sleep(0.01)  # Simulate query execution
        return True
    
    async def cleanup_idle_connections(self):
        """Nettoie les connexions inactives"""
        now = datetime.now(timezone.utc)
        idle_timeout = timedelta(seconds=self.pool_config.idle_timeout)
        
        to_remove = []
        for connection_id, metrics in self._connection_metrics.items():
            if (self._connection_states[connection_id] == ConnectionState.IDLE and
                now - metrics.last_used > idle_timeout and
                self._total_connections > self.config.min_size):
                to_remove.append(connection_id)
        
        for connection_id in to_remove:
            await self._close_connection(connection_id)
            self.logger.debug(f"Closed idle connection {connection_id}")
    
    async def _close_connection(self, connection_id: str):
        """Ferme une connexion"""
        if connection_id in self._connections:
            try:
                connection = self._connections[connection_id]
                await self._close_database_connection(connection)
            except Exception as e:
                self.logger.warning(f"Error closing connection {connection_id}: {e}")
            finally:
                # Clean up tracking
                del self._connections[connection_id]
                del self._connection_states[connection_id]
                del self._connection_metrics[connection_id]
                if connection_id in self._connection_weights:
                    del self._connection_weights[connection_id]
                
                self._total_connections -= 1
    
    async def _close_database_connection(self, connection: Any):
        """Ferme une connexion database réelle"""
        # Mock implementation
        await asyncio.sleep(0.01)
    
    async def close_all_connections(self):
        """Ferme toutes les connexions"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        connection_ids = list(self._connections.keys())
        for connection_id in connection_ids:
            await self._close_connection(connection_id)
        
        self.logger.info(f"Closed all connections for pool {self.pool_id}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du pool"""
        return {
            "pool_id": self.pool_id,
            "total_connections": self._total_connections,
            "active_connections": self._active_connections,
            "idle_connections": len([
                s for s in self._connection_states.values() 
                if s == ConnectionState.IDLE
            ]),
            "error_connections": len([
                s for s in self._connection_states.values() 
                if s == ConnectionState.ERROR
            ]),
            "is_healthy": self._is_healthy,
            "circuit_breaker_state": self._circuit_breaker_state,
            "failure_count": self._failure_count,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "performance_stats": dict(self._performance_stats),
            "config": {
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
                "tenant_id": self.config.tenant_id,
                "connection_type": self.config.connection_type.value
            }
        }


class ConnectionManager:
    """
    Gestionnaire de connexions ultra-avancé pour l'isolation multi-tenant
    
    Features:
    - Pool de connexions par tenant/database
    - Load balancing intelligent
    - Circuit breaker pattern
    - Health monitoring automatique
    - Failover automatique
    - Métriques de performance détaillées
    - Nettoyage automatique des connexions inactives
    - Support multi-database
    """
    
    def __init__(self, pool_config: Optional[PoolConfiguration] = None):
        self.pool_config = pool_config or PoolConfiguration()
        self.logger = logging.getLogger("connection_manager")
        
        # Pool management
        self._pools: Dict[str, ConnectionPool] = {}
        self._pool_configs: Dict[str, DatabaseConnectionConfig] = {}
        
        # Global state
        self._is_initialized = False
        self._total_connections = 0
        self._shutdown_event = asyncio.Event()
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Tenant mappings
        self._tenant_pools: Dict[str, Set[str]] = {}  # tenant_id -> pool_ids
        self._database_pools: Dict[str, str] = {}     # database_name -> pool_id
        
        # Performance tracking
        self._global_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "pools_created": 0,
            "connections_created": 0,
            "failovers_executed": 0
        }
        
        # Weak references for cleanup
        self._pool_refs = weakref.WeakSet()
    
    async def initialize(self):
        """Initialise le gestionnaire de connexions"""
        try:
            if self._is_initialized:
                return
            
            # Start monitoring tasks
            if self.pool_config.enable_metrics:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._is_initialized = True
            self.logger.info("Connection manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection manager: {e}")
            raise ConnectionError(f"Connection manager initialization failed: {e}")
    
    def register_database_config(
        self, 
        config: DatabaseConnectionConfig,
        pool_id: Optional[str] = None
    ) -> str:
        """Enregistre une configuration de base de données"""
        if not pool_id:
            pool_id = self._generate_pool_id(config)
        
        self._pool_configs[pool_id] = config
        
        # Track tenant mapping
        if config.tenant_id:
            if config.tenant_id not in self._tenant_pools:
                self._tenant_pools[config.tenant_id] = set()
            self._tenant_pools[config.tenant_id].add(pool_id)
        
        # Track database mapping
        self._database_pools[config.database] = pool_id
        
        self.logger.info(f"Registered database config for pool {pool_id}")
        return pool_id
    
    def _generate_pool_id(self, config: DatabaseConnectionConfig) -> str:
        """Génère un ID unique pour le pool"""
        components = [
            config.host,
            str(config.port),
            config.database,
            config.tenant_id or "shared",
            config.connection_type.value
        ]
        
        hash_input = "_".join(components)
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        return f"pool_{hash_value}"
    
    async def get_pool(self, pool_id: str) -> ConnectionPool:
        """Obtient ou crée un pool de connexions"""
        if pool_id not in self._pools:
            await self._create_pool(pool_id)
        
        return self._pools[pool_id]
    
    async def _create_pool(self, pool_id: str):
        """Crée un nouveau pool de connexions"""
        if pool_id not in self._pool_configs:
            raise ConnectionError(f"No configuration found for pool {pool_id}")
        
        config = self._pool_configs[pool_id]
        
        # Check global connection limit
        if self._total_connections >= self.pool_config.max_total_connections:
            raise ConnectionError("Global connection limit reached")
        
        try:
            pool = ConnectionPool(pool_id, config, self.pool_config)
            await pool.initialize()
            
            self._pools[pool_id] = pool
            self._pool_refs.add(pool)
            
            self._total_connections += config.min_size
            self._global_stats["pools_created"] += 1
            self._global_stats["connections_created"] += config.min_size
            
            self.logger.info(f"Created pool {pool_id} with {config.min_size} connections")
            
        except Exception as e:
            self.logger.error(f"Failed to create pool {pool_id}: {e}")
            raise ConnectionError(f"Pool creation failed: {e}")
    
    @asynccontextmanager
    async def get_connection(
        self, 
        context: TenantContext,
        connection_type: ConnectionType = ConnectionType.READ,
        database_name: Optional[str] = None
    ):
        """Obtient une connexion pour un tenant"""
        pool_id = None
        try:
            # Determine pool ID
            pool_id = await self._resolve_pool_id(context, connection_type, database_name)
            
            if not pool_id:
                raise ConnectionError(f"No pool available for tenant {context.tenant_id}")
            
            # Get pool and acquire connection
            pool = await self.get_pool(pool_id)
            
            start_time = datetime.now(timezone.utc)
            
            async with pool.acquire_connection(connection_type) as connection:
                # Track usage
                response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                self.logger.debug(
                    f"Acquired connection from pool {pool_id} for tenant {context.tenant_id} "
                    f"(type: {connection_type.value}, time: {response_time:.2f}ms)"
                )
                
                yield connection
                
                # Record success
                self._global_stats["successful_queries"] += 1
                
        except Exception as e:
            self._global_stats["failed_queries"] += 1
            self.logger.error(f"Failed to get connection for tenant {context.tenant_id}: {e}")
            
            # Try failover if enabled
            if self.pool_config.enable_automatic_failover and pool_id:
                async with self._try_failover(context, connection_type, database_name, pool_id) as fallback_conn:
                    if fallback_conn:
                        yield fallback_conn
                        return
            
            raise ConnectionError(f"Connection acquisition failed: {e}")
    
    async def _resolve_pool_id(
        self, 
        context: TenantContext,
        connection_type: ConnectionType,
        database_name: Optional[str]
    ) -> Optional[str]:
        """Résout l'ID du pool pour un tenant"""
        # Try tenant-specific pools first
        if context.tenant_id in self._tenant_pools:
            tenant_pools = self._tenant_pools[context.tenant_id]
            
            # Filter by connection type and database
            suitable_pools = []
            for pool_id in tenant_pools:
                config = self._pool_configs[pool_id]
                if (config.connection_type == connection_type and
                    (not database_name or config.database == database_name)):
                    suitable_pools.append(pool_id)
            
            if suitable_pools:
                # Return pool with best health/performance
                return await self._select_best_pool(suitable_pools)
        
        # Try database-specific pools
        if database_name and database_name in self._database_pools:
            return self._database_pools[database_name]
        
        # Fallback to any suitable pool
        for pool_id, config in self._pool_configs.items():
            if config.connection_type == connection_type:
                return pool_id
        
        return None
    
    async def _select_best_pool(self, pool_ids: List[str]) -> str:
        """Sélectionne le meilleur pool parmi les candidats"""
        best_pool_id = None
        best_score = -1
        
        for pool_id in pool_ids:
            if pool_id in self._pools:
                pool = self._pools[pool_id]
                stats = pool.get_pool_stats()
                
                # Calculate pool score based on health, performance, and load
                score = self._calculate_pool_score(stats)
                
                if score > best_score:
                    best_score = score
                    best_pool_id = pool_id
        
        return best_pool_id or pool_ids[0]  # Fallback to first pool
    
    def _calculate_pool_score(self, stats: Dict[str, Any]) -> float:
        """Calcule le score d'un pool"""
        if not stats["is_healthy"]:
            return 0.0
        
        # Health factor (40%)
        health_score = 1.0 if stats["is_healthy"] else 0.0
        
        # Load factor (30%)
        load_ratio = stats["active_connections"] / max(stats["total_connections"], 1)
        load_score = 1.0 - load_ratio
        
        # Performance factor (30%)
        perf_stats = stats["performance_stats"]
        success_rate = (
            perf_stats["successful_queries"] / 
            max(perf_stats["total_queries"], 1)
        )
        
        # Combine scores
        total_score = (
            health_score * 0.4 +
            load_score * 0.3 +
            success_rate * 0.3
        )
        
        return total_score
    
    @asynccontextmanager
    async def _try_failover(
        self, 
        context: TenantContext,
        connection_type: ConnectionType,
        database_name: Optional[str],
        failed_pool_id: str
    ):
        """Tente un failover vers un autre pool"""
        self.logger.warning(f"Attempting failover from pool {failed_pool_id}")
        
        # Find alternative pools
        alternative_pools = []
        for pool_id, config in self._pool_configs.items():
            if (pool_id != failed_pool_id and
                config.connection_type == connection_type and
                (not database_name or config.database == database_name)):
                alternative_pools.append(pool_id)
        
        for attempt, pool_id in enumerate(alternative_pools):
            if attempt >= self.pool_config.max_failover_attempts:
                break
            
            try:
                pool = await self.get_pool(pool_id)
                async with pool.acquire_connection(connection_type) as connection:
                    self.logger.info(f"Failover successful to pool {pool_id}")
                    self._global_stats["failovers_executed"] += 1
                    yield connection
                    return
                    
            except Exception as e:
                self.logger.warning(f"Failover attempt {attempt + 1} to pool {pool_id} failed: {e}")
                continue
        
        self.logger.error("All failover attempts failed")
        yield None
    
    async def _monitoring_loop(self):
        """Boucle de monitoring"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Monitor every minute
                await self._collect_and_log_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    async def _collect_and_log_metrics(self):
        """Collecte et log les métriques"""
        total_pools = len(self._pools)
        healthy_pools = sum(1 for pool in self._pools.values() if pool._is_healthy)
        
        total_connections = sum(pool._total_connections for pool in self._pools.values())
        active_connections = sum(pool._active_connections for pool in self._pools.values())
        
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_pools": total_pools,
            "healthy_pools": healthy_pools,
            "total_connections": total_connections,
            "active_connections": active_connections,
            "global_stats": dict(self._global_stats)
        }
        
        self.logger.info(f"Connection manager metrics: {json.dumps(metrics, indent=2)}")
    
    async def _cleanup_loop(self):
        """Boucle de nettoyage"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_idle_resources()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_idle_resources(self):
        """Nettoie les ressources inactives"""
        for pool in self._pools.values():
            try:
                await pool.cleanup_idle_connections()
            except Exception as e:
                self.logger.warning(f"Error cleaning up pool {pool.pool_id}: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification de santé globale"""
        pool_health = {}
        for pool_id, pool in self._pools.items():
            pool_health[pool_id] = {
                "healthy": pool._is_healthy,
                "stats": pool.get_pool_stats()
            }
        
        total_pools = len(self._pools)
        healthy_pools = sum(1 for pool in self._pools.values() if pool._is_healthy)
        
        return {
            "connection_manager_healthy": healthy_pools > 0,
            "total_pools": total_pools,
            "healthy_pools": healthy_pools,
            "health_ratio": healthy_pools / max(total_pools, 1),
            "pools": pool_health,
            "global_stats": dict(self._global_stats)
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Obtient les statistiques détaillées"""
        pool_stats = {}
        for pool_id, pool in self._pools.items():
            pool_stats[pool_id] = pool.get_pool_stats()
        
        return {
            "connection_manager": {
                "total_pools": len(self._pools),
                "total_connections": sum(p._total_connections for p in self._pools.values()),
                "active_connections": sum(p._active_connections for p in self._pools.values()),
                "global_stats": dict(self._global_stats)
            },
            "pool_statistics": pool_stats,
            "tenant_mappings": {
                tenant_id: list(pools) 
                for tenant_id, pools in self._tenant_pools.items()
            },
            "database_mappings": dict(self._database_pools)
        }
    
    async def shutdown(self):
        """Arrêt propre du gestionnaire"""
        self.logger.info("Shutting down connection manager...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel monitoring tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all pools
        for pool in self._pools.values():
            try:
                await pool.close_all_connections()
            except Exception as e:
                self.logger.warning(f"Error closing pool {pool.pool_id}: {e}")
        
        self._pools.clear()
        self._pool_configs.clear()
        self._tenant_pools.clear()
        self._database_pools.clear()
        
        self.logger.info("Connection manager shutdown completed")


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


async def get_connection_manager() -> ConnectionManager:
    """Obtient l'instance globale du gestionnaire de connexions"""
    global _connection_manager
    if not _connection_manager:
        _connection_manager = ConnectionManager()
        await _connection_manager.initialize()
    return _connection_manager


async def shutdown_connection_manager():
    """Arrête l'instance globale du gestionnaire de connexions"""
    global _connection_manager
    if _connection_manager:
        await _connection_manager.shutdown()
        _connection_manager = None
