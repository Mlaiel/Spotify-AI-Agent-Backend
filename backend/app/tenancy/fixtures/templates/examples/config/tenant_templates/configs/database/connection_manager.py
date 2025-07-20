"""
Advanced Connection Manager for Multi-Database Enterprise Architecture
=====================================================================

This module provides a sophisticated connection management system that handles
multiple database types with advanced features like load balancing, failover,
health monitoring, and AI-powered optimization.

Features:
- Multi-database type support (PostgreSQL, MongoDB, Redis, ClickHouse, TimescaleDB, Elasticsearch)
- Intelligent connection pooling with auto-scaling
- Load balancing across multiple instances
- Automatic failover and health monitoring
- Circuit breaker pattern implementation
- Connection metrics and performance tracking
- Tenant-aware connection management
- Security hardening and encryption
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Callable
import json
import hashlib
from datetime import datetime, timedelta

import asyncpg
import motor.motor_asyncio
import aioredis
import asyncpg_pool
from elasticsearch import AsyncElasticsearch
import aioch
import ssl
import certifi

from . import DatabaseType, config_loader


class ConnectionState(Enum):
    """Connection state enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


class LoadBalancingStrategy(Enum):
    """Load balancing strategy enumeration"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    HEALTH_BASED = "health_based"


@dataclass
class ConnectionMetrics:
    """Connection metrics tracking"""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    average_response_time: float = 0.0
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0
    
    def calculate_success_rate(self) -> float:
        """Calculate connection success rate"""
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0.0
    
    def calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        success_rate = self.calculate_success_rate()
        response_time_score = max(0, 100 - (self.average_response_time * 10))
        return (success_rate * 0.7) + (response_time_score * 0.3)


@dataclass
class DatabaseInstance:
    """Database instance configuration"""
    host: str
    port: int
    database: str
    username: str = ""
    password: str = ""
    weight: int = 1
    max_connections: int = 20
    timeout: int = 30
    ssl_config: Optional[Dict[str, Any]] = None
    
    # Runtime state
    state: ConnectionState = ConnectionState.HEALTHY
    metrics: ConnectionMetrics = field(default_factory=ConnectionMetrics)
    pool: Optional[Any] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    
    def get_connection_string(self, db_type: DatabaseType) -> str:
        """Generate connection string for the database type"""
        if db_type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif db_type == DatabaseType.MONGODB:
            auth_part = f"{self.username}:{self.password}@" if self.username else ""
            return f"mongodb://{auth_part}{self.host}:{self.port}/{self.database}"
        elif db_type == DatabaseType.REDIS:
            auth_part = f":{self.password}@" if self.password else ""
            return f"redis://{auth_part}{self.host}:{self.port}/0"
        elif db_type == DatabaseType.CLICKHOUSE:
            return f"clickhouse://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif db_type == DatabaseType.TIMESCALEDB:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif db_type == DatabaseType.ELASTICSEARCH:
            scheme = "https" if self.ssl_config else "http"
            auth_part = f"{self.username}:{self.password}@" if self.username else ""
            return f"{scheme}://{auth_part}{self.host}:{self.port}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")


class CircuitBreaker:
    """Circuit breaker implementation for connection resilience"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class ConnectionManager:
    """
    Advanced connection manager for multi-database enterprise architecture
    """
    
    def __init__(self, config: Dict[str, Any], tenant_id: Optional[str] = None):
        self.config = config
        self.tenant_id = tenant_id
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.instances: Dict[str, DatabaseInstance] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.load_balancer = LoadBalancer()
        self.health_monitor = HealthMonitor(self)
        
        # State management
        self.is_initialized = False
        self.metrics_collector = MetricsCollector()
        
        # AI optimization components
        self.query_optimizer = QueryOptimizer()
        self.cache_manager = CacheManager()
        
    async def initialize(self):
        """Initialize connection manager and all database instances"""
        if self.is_initialized:
            return
        
        try:
            await self._discover_instances()
            await self._initialize_pools()
            await self._start_health_monitoring()
            
            self.is_initialized = True
            self.logger.info(f"Connection manager initialized with {len(self.instances)} instances")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connection manager: {e}")
            raise
    
    async def _discover_instances(self):
        """Discover and configure database instances from config"""
        # Parse database configuration based on type
        for db_type in DatabaseType:
            if db_type.value in self.config:
                db_config = self.config[db_type.value]
                await self._configure_instances_for_type(db_type, db_config)
    
    async def _configure_instances_for_type(self, db_type: DatabaseType, config: Dict[str, Any]):
        """Configure instances for a specific database type"""
        connection_config = config.get('connection', {})
        
        if 'cluster' in config and config['cluster'].get('enabled', False):
            # Multi-instance cluster configuration
            await self._configure_cluster_instances(db_type, config)
        else:
            # Single instance configuration
            instance = DatabaseInstance(
                host=connection_config.get('host', 'localhost'),
                port=connection_config.get('port', self._get_default_port(db_type)),
                database=connection_config.get('database', 'default'),
                username=connection_config.get('username', ''),
                password=connection_config.get('password', ''),
                max_connections=connection_config.get('pool', {}).get('max_connections', 20),
                timeout=connection_config.get('pool', {}).get('timeout', 30),
                ssl_config=connection_config.get('ssl', {})
            )
            
            instance_key = f"{db_type.value}_primary"
            self.instances[instance_key] = instance
            self.circuit_breakers[instance_key] = CircuitBreaker()
    
    async def _configure_cluster_instances(self, db_type: DatabaseType, config: Dict[str, Any]):
        """Configure multiple instances for clustered setup"""
        cluster_config = config['cluster']
        
        if db_type == DatabaseType.ELASTICSEARCH:
            # Elasticsearch cluster configuration
            hosts = config['connection'].get('hosts', [])
            for i, host_config in enumerate(hosts):
                instance = DatabaseInstance(
                    host=host_config['host'],
                    port=host_config['port'],
                    database="",  # Elasticsearch doesn't use database in connection
                    username=config['connection']['authentication'].get('username', ''),
                    password=config['connection']['authentication'].get('password', ''),
                    ssl_config=host_config.get('ssl_config', {})
                )
                
                instance_key = f"{db_type.value}_node_{i}"
                self.instances[instance_key] = instance
                self.circuit_breakers[instance_key] = CircuitBreaker()
        
        elif db_type == DatabaseType.CLICKHOUSE:
            # ClickHouse cluster configuration
            shards = cluster_config.get('shards', [])
            for shard_idx, shard in enumerate(shards):
                for replica_idx, replica in enumerate(shard.get('replicas', [])):
                    instance = DatabaseInstance(
                        host=replica['host'],
                        port=replica['port'],
                        database=config['connection'].get('database', 'default'),
                        username=config['connection'].get('username', 'default'),
                        password=config['connection'].get('password', ''),
                        weight=replica.get('weight', 1)
                    )
                    
                    instance_key = f"{db_type.value}_shard_{shard_idx}_replica_{replica_idx}"
                    self.instances[instance_key] = instance
                    self.circuit_breakers[instance_key] = CircuitBreaker()
    
    def _get_default_port(self, db_type: DatabaseType) -> int:
        """Get default port for database type"""
        defaults = {
            DatabaseType.POSTGRESQL: 5432,
            DatabaseType.MONGODB: 27017,
            DatabaseType.REDIS: 6379,
            DatabaseType.CLICKHOUSE: 9000,
            DatabaseType.TIMESCALEDB: 5432,
            DatabaseType.ELASTICSEARCH: 9200
        }
        return defaults.get(db_type, 5432)
    
    async def _initialize_pools(self):
        """Initialize connection pools for all instances"""
        initialization_tasks = []
        
        for instance_key, instance in self.instances.items():
            task = self._initialize_instance_pool(instance_key, instance)
            initialization_tasks.append(task)
        
        # Initialize pools concurrently
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        # Handle initialization results
        for i, result in enumerate(results):
            instance_key = list(self.instances.keys())[i]
            if isinstance(result, Exception):
                self.logger.error(f"Failed to initialize pool for {instance_key}: {result}")
                self.instances[instance_key].state = ConnectionState.UNHEALTHY
            else:
                self.logger.info(f"Successfully initialized pool for {instance_key}")
    
    async def _initialize_instance_pool(self, instance_key: str, instance: DatabaseInstance):
        """Initialize connection pool for a specific instance"""
        db_type = DatabaseType(instance_key.split('_')[0])
        
        try:
            if db_type in [DatabaseType.POSTGRESQL, DatabaseType.TIMESCALEDB]:
                instance.pool = await self._create_postgresql_pool(instance)
            elif db_type == DatabaseType.MONGODB:
                instance.pool = await self._create_mongodb_pool(instance)
            elif db_type == DatabaseType.REDIS:
                instance.pool = await self._create_redis_pool(instance)
            elif db_type == DatabaseType.CLICKHOUSE:
                instance.pool = await self._create_clickhouse_pool(instance)
            elif db_type == DatabaseType.ELASTICSEARCH:
                instance.pool = await self._create_elasticsearch_pool(instance)
            
            instance.state = ConnectionState.HEALTHY
            
        except Exception as e:
            instance.state = ConnectionState.UNHEALTHY
            instance.last_failure = datetime.now()
            instance.consecutive_failures += 1
            raise e
    
    async def _create_postgresql_pool(self, instance: DatabaseInstance):
        """Create PostgreSQL connection pool"""
        ssl_context = None
        if instance.ssl_config and instance.ssl_config.get('enabled', False):
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            if instance.ssl_config.get('cert_file'):
                ssl_context.load_cert_chain(
                    instance.ssl_config['cert_file'],
                    instance.ssl_config.get('key_file')
                )
        
        return await asyncpg.create_pool(
            host=instance.host,
            port=instance.port,
            database=instance.database,
            user=instance.username,
            password=instance.password,
            min_size=5,
            max_size=instance.max_connections,
            command_timeout=instance.timeout,
            ssl=ssl_context
        )
    
    async def _create_mongodb_pool(self, instance: DatabaseInstance):
        """Create MongoDB connection pool"""
        connection_string = instance.get_connection_string(DatabaseType.MONGODB)
        client = motor.motor_asyncio.AsyncIOMotorClient(
            connection_string,
            maxPoolSize=instance.max_connections,
            minPoolSize=5,
            maxIdleTimeMS=30000,
            waitQueueTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000,
            serverSelectionTimeoutMS=5000
        )
        return client[instance.database]
    
    async def _create_redis_pool(self, instance: DatabaseInstance):
        """Create Redis connection pool"""
        return aioredis.ConnectionPool.from_url(
            instance.get_connection_string(DatabaseType.REDIS),
            max_connections=instance.max_connections,
            retry_on_timeout=True,
            socket_keepalive=True,
            health_check_interval=30
        )
    
    async def _create_clickhouse_pool(self, instance: DatabaseInstance):
        """Create ClickHouse connection pool"""
        return await aioch.Pool(
            host=instance.host,
            port=instance.port,
            database=instance.database,
            user=instance.username,
            password=instance.password,
            minsize=5,
            maxsize=instance.max_connections,
            loop=asyncio.get_event_loop()
        )
    
    async def _create_elasticsearch_pool(self, instance: DatabaseInstance):
        """Create Elasticsearch connection pool"""
        return AsyncElasticsearch(
            hosts=[{
                'host': instance.host,
                'port': instance.port,
                'use_ssl': bool(instance.ssl_config),
                'verify_certs': instance.ssl_config.get('verify_ssl_cert', True) if instance.ssl_config else False
            }],
            http_auth=(instance.username, instance.password) if instance.username else None,
            timeout=instance.timeout,
            max_retries=3,
            retry_on_timeout=True
        )
    
    @asynccontextmanager
    async def get_connection(self, 
                           db_type: DatabaseType,
                           strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_BASED,
                           tenant_context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Any, None]:
        """
        Get a database connection with load balancing and failover
        
        Args:
            db_type: Type of database connection needed
            strategy: Load balancing strategy to use
            tenant_context: Additional tenant-specific context
            
        Yields:
            Database connection object
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Select best instance using load balancing strategy
        instance_key = await self.load_balancer.select_instance(
            self.instances, db_type, strategy
        )
        
        if not instance_key:
            raise Exception(f"No healthy instances available for {db_type.value}")
        
        instance = self.instances[instance_key]
        circuit_breaker = self.circuit_breakers[instance_key]
        
        start_time = time.time()
        connection = None
        
        try:
            # Get connection through circuit breaker
            connection = await circuit_breaker.call(
                self._acquire_connection, instance, db_type
            )
            
            # Apply tenant context if needed
            if tenant_context and self.tenant_id:
                await self._apply_tenant_context(connection, db_type, tenant_context)
            
            # Update metrics
            response_time = time.time() - start_time
            instance.metrics.average_response_time = (
                (instance.metrics.average_response_time + response_time) / 2
            )
            instance.metrics.success_count += 1
            instance.metrics.active_connections += 1
            
            yield connection
            
        except Exception as e:
            # Update error metrics
            instance.metrics.error_count += 1
            instance.metrics.failed_connections += 1
            instance.consecutive_failures += 1
            instance.last_failure = datetime.now()
            
            # Update instance state based on consecutive failures
            if instance.consecutive_failures >= 3:
                instance.state = ConnectionState.UNHEALTHY
            elif instance.consecutive_failures >= 1:
                instance.state = ConnectionState.DEGRADED
            
            self.logger.error(f"Connection error for {instance_key}: {e}")
            raise
        
        finally:
            # Clean up connection
            if connection:
                await self._release_connection(connection, instance, db_type)
                instance.metrics.active_connections -= 1
    
    async def _acquire_connection(self, instance: DatabaseInstance, db_type: DatabaseType):
        """Acquire connection from instance pool"""
        if not instance.pool:
            raise Exception(f"Pool not initialized for instance")
        
        if db_type in [DatabaseType.POSTGRESQL, DatabaseType.TIMESCALEDB]:
            return instance.pool.acquire()
        elif db_type == DatabaseType.MONGODB:
            return instance.pool  # MongoDB client acts as connection
        elif db_type == DatabaseType.REDIS:
            return aioredis.Redis(connection_pool=instance.pool)
        elif db_type == DatabaseType.CLICKHOUSE:
            return await instance.pool.acquire()
        elif db_type == DatabaseType.ELASTICSEARCH:
            return instance.pool  # Elasticsearch client acts as connection
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    async def _release_connection(self, connection: Any, instance: DatabaseInstance, db_type: DatabaseType):
        """Release connection back to pool"""
        try:
            if db_type in [DatabaseType.POSTGRESQL, DatabaseType.TIMESCALEDB]:
                await connection.close() if hasattr(connection, 'close') else None
            elif db_type == DatabaseType.CLICKHOUSE:
                await instance.pool.release(connection)
            elif db_type == DatabaseType.REDIS:
                await connection.close()
            # MongoDB and Elasticsearch connections are managed by their clients
            
        except Exception as e:
            self.logger.warning(f"Error releasing connection: {e}")
    
    async def _apply_tenant_context(self, connection: Any, db_type: DatabaseType, context: Dict[str, Any]):
        """Apply tenant-specific context to connection"""
        if db_type in [DatabaseType.POSTGRESQL, DatabaseType.TIMESCALEDB]:
            # Set tenant context in PostgreSQL session
            await connection.execute(
                "SELECT set_config('app.current_tenant', $1, false)",
                self.tenant_id
            )
        elif db_type == DatabaseType.MONGODB:
            # MongoDB tenant context handled at query level
            pass
        elif db_type == DatabaseType.REDIS:
            # Redis tenant context via key prefixing
            pass
    
    async def _start_health_monitoring(self):
        """Start background health monitoring"""
        asyncio.create_task(self.health_monitor.start_monitoring())
    
    async def close(self):
        """Close all connections and cleanup resources"""
        self.logger.info("Closing connection manager...")
        
        # Stop health monitoring
        await self.health_monitor.stop_monitoring()
        
        # Close all pools
        close_tasks = []
        for instance_key, instance in self.instances.items():
            if instance.pool:
                task = self._close_instance_pool(instance_key, instance)
                close_tasks.append(task)
        
        await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self.is_initialized = False
        self.logger.info("Connection manager closed")
    
    async def _close_instance_pool(self, instance_key: str, instance: DatabaseInstance):
        """Close connection pool for specific instance"""
        db_type = DatabaseType(instance_key.split('_')[0])
        
        try:
            if db_type in [DatabaseType.POSTGRESQL, DatabaseType.TIMESCALEDB]:
                await instance.pool.close()
            elif db_type == DatabaseType.MONGODB:
                instance.pool.client.close()
            elif db_type == DatabaseType.REDIS:
                await instance.pool.disconnect()
            elif db_type == DatabaseType.CLICKHOUSE:
                instance.pool.close()
                await instance.pool.wait_closed()
            elif db_type == DatabaseType.ELASTICSEARCH:
                await instance.pool.close()
            
        except Exception as e:
            self.logger.error(f"Error closing pool for {instance_key}: {e}")


class LoadBalancer:
    """Intelligent load balancer for database connections"""
    
    def __init__(self):
        self.round_robin_counters: Dict[str, int] = {}
    
    async def select_instance(self, 
                            instances: Dict[str, DatabaseInstance],
                            db_type: DatabaseType,
                            strategy: LoadBalancingStrategy) -> Optional[str]:
        """Select best instance based on strategy"""
        
        # Filter healthy instances for the specified database type
        healthy_instances = {
            key: instance for key, instance in instances.items()
            if (instance.state in [ConnectionState.HEALTHY, ConnectionState.DEGRADED] and
                key.startswith(db_type.value))
        }
        
        if not healthy_instances:
            return None
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_instances, db_type)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_instances)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_instances, db_type)
        elif strategy == LoadBalancingStrategy.RANDOM:
            return self._random_selection(healthy_instances)
        elif strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based_selection(healthy_instances)
        else:
            return list(healthy_instances.keys())[0]  # Default to first healthy
    
    def _round_robin_selection(self, instances: Dict[str, DatabaseInstance], db_type: DatabaseType) -> str:
        """Round-robin instance selection"""
        counter_key = db_type.value
        if counter_key not in self.round_robin_counters:
            self.round_robin_counters[counter_key] = 0
        
        instance_keys = list(instances.keys())
        selected_index = self.round_robin_counters[counter_key] % len(instance_keys)
        self.round_robin_counters[counter_key] += 1
        
        return instance_keys[selected_index]
    
    def _least_connections_selection(self, instances: Dict[str, DatabaseInstance]) -> str:
        """Select instance with least active connections"""
        return min(instances.keys(), 
                  key=lambda k: instances[k].metrics.active_connections)
    
    def _weighted_round_robin_selection(self, instances: Dict[str, DatabaseInstance], db_type: DatabaseType) -> str:
        """Weighted round-robin based on instance weights"""
        # Create weighted list of instances
        weighted_instances = []
        for key, instance in instances.items():
            weighted_instances.extend([key] * instance.weight)
        
        if not weighted_instances:
            return list(instances.keys())[0]
        
        counter_key = f"{db_type.value}_weighted"
        if counter_key not in self.round_robin_counters:
            self.round_robin_counters[counter_key] = 0
        
        selected_index = self.round_robin_counters[counter_key] % len(weighted_instances)
        self.round_robin_counters[counter_key] += 1
        
        return weighted_instances[selected_index]
    
    def _random_selection(self, instances: Dict[str, DatabaseInstance]) -> str:
        """Random instance selection"""
        import random
        return random.choice(list(instances.keys()))
    
    def _health_based_selection(self, instances: Dict[str, DatabaseInstance]) -> str:
        """Select instance based on health score"""
        return max(instances.keys(),
                  key=lambda k: instances[k].metrics.calculate_health_score())


class HealthMonitor:
    """Health monitoring system for database instances"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        self.check_interval = 30  # seconds
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def start_monitoring(self):
        """Start health monitoring background task"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._check_all_instances()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_instances(self):
        """Check health of all instances"""
        check_tasks = []
        
        for instance_key, instance in self.connection_manager.instances.items():
            task = self._check_instance_health(instance_key, instance)
            check_tasks.append(task)
        
        await asyncio.gather(*check_tasks, return_exceptions=True)
    
    async def _check_instance_health(self, instance_key: str, instance: DatabaseInstance):
        """Check health of a specific instance"""
        db_type = DatabaseType(instance_key.split('_')[0])
        
        try:
            start_time = time.time()
            
            # Perform database-specific health check
            if db_type in [DatabaseType.POSTGRESQL, DatabaseType.TIMESCALEDB]:
                await self._check_postgresql_health(instance)
            elif db_type == DatabaseType.MONGODB:
                await self._check_mongodb_health(instance)
            elif db_type == DatabaseType.REDIS:
                await self._check_redis_health(instance)
            elif db_type == DatabaseType.CLICKHOUSE:
                await self._check_clickhouse_health(instance)
            elif db_type == DatabaseType.ELASTICSEARCH:
                await self._check_elasticsearch_health(instance)
            
            # Update health metrics
            response_time = time.time() - start_time
            instance.metrics.last_health_check = datetime.now()
            instance.metrics.average_response_time = (
                (instance.metrics.average_response_time + response_time) / 2
            )
            
            # Update instance state based on consecutive failures
            if instance.consecutive_failures > 0:
                instance.consecutive_failures = max(0, instance.consecutive_failures - 1)
            
            if instance.consecutive_failures == 0:
                instance.state = ConnectionState.HEALTHY
            
        except Exception as e:
            instance.consecutive_failures += 1
            instance.last_failure = datetime.now()
            
            if instance.consecutive_failures >= 5:
                instance.state = ConnectionState.UNHEALTHY
            elif instance.consecutive_failures >= 2:
                instance.state = ConnectionState.DEGRADED
            
            self.logger.warning(f"Health check failed for {instance_key}: {e}")
    
    async def _check_postgresql_health(self, instance: DatabaseInstance):
        """PostgreSQL specific health check"""
        if instance.pool:
            async with instance.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
    
    async def _check_mongodb_health(self, instance: DatabaseInstance):
        """MongoDB specific health check"""
        if instance.pool:
            await instance.pool.client.admin.command("ping")
    
    async def _check_redis_health(self, instance: DatabaseInstance):
        """Redis specific health check"""
        if instance.pool:
            redis = aioredis.Redis(connection_pool=instance.pool)
            await redis.ping()
            await redis.close()
    
    async def _check_clickhouse_health(self, instance: DatabaseInstance):
        """ClickHouse specific health check"""
        if instance.pool:
            async with instance.pool.acquire() as conn:
                await conn.execute("SELECT 1")
    
    async def _check_elasticsearch_health(self, instance: DatabaseInstance):
        """Elasticsearch specific health check"""
        if instance.pool:
            await instance.pool.cluster.health()


class MetricsCollector:
    """Metrics collection and aggregation system"""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    def collect_metrics(self, instances: Dict[str, DatabaseInstance]) -> Dict[str, Any]:
        """Collect current metrics from all instances"""
        timestamp = datetime.now()
        aggregated_metrics = {
            'timestamp': timestamp.isoformat(),
            'instances': {},
            'totals': {
                'total_instances': len(instances),
                'healthy_instances': 0,
                'degraded_instances': 0,
                'unhealthy_instances': 0,
                'total_connections': 0,
                'active_connections': 0,
                'average_response_time': 0.0,
                'overall_success_rate': 0.0
            }
        }
        
        total_response_time = 0.0
        total_success_rates = []
        
        for instance_key, instance in instances.items():
            instance_metrics = {
                'state': instance.state.value,
                'metrics': {
                    'total_connections': instance.metrics.total_connections,
                    'active_connections': instance.metrics.active_connections,
                    'failed_connections': instance.metrics.failed_connections,
                    'average_response_time': instance.metrics.average_response_time,
                    'success_rate': instance.metrics.calculate_success_rate(),
                    'health_score': instance.metrics.calculate_health_score(),
                    'last_health_check': instance.metrics.last_health_check.isoformat() if instance.metrics.last_health_check else None
                },
                'consecutive_failures': instance.consecutive_failures,
                'last_failure': instance.last_failure.isoformat() if instance.last_failure else None
            }
            
            aggregated_metrics['instances'][instance_key] = instance_metrics
            
            # Update totals
            if instance.state == ConnectionState.HEALTHY:
                aggregated_metrics['totals']['healthy_instances'] += 1
            elif instance.state == ConnectionState.DEGRADED:
                aggregated_metrics['totals']['degraded_instances'] += 1
            elif instance.state == ConnectionState.UNHEALTHY:
                aggregated_metrics['totals']['unhealthy_instances'] += 1
            
            aggregated_metrics['totals']['total_connections'] += instance.metrics.total_connections
            aggregated_metrics['totals']['active_connections'] += instance.metrics.active_connections
            
            total_response_time += instance.metrics.average_response_time
            total_success_rates.append(instance.metrics.calculate_success_rate())
        
        # Calculate averages
        if instances:
            aggregated_metrics['totals']['average_response_time'] = total_response_time / len(instances)
            aggregated_metrics['totals']['overall_success_rate'] = sum(total_success_rates) / len(total_success_rates)
        
        # Store in history
        self.metrics_history.append(aggregated_metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
        
        return aggregated_metrics
    
    def get_metrics_summary(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for specified time range"""
        cutoff_time = datetime.now() - timedelta(minutes=time_range_minutes)
        
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate summary statistics
        summary = {
            'time_range_minutes': time_range_minutes,
            'data_points': len(recent_metrics),
            'average_response_time': {
                'min': min(m['totals']['average_response_time'] for m in recent_metrics),
                'max': max(m['totals']['average_response_time'] for m in recent_metrics),
                'avg': sum(m['totals']['average_response_time'] for m in recent_metrics) / len(recent_metrics)
            },
            'success_rate': {
                'min': min(m['totals']['overall_success_rate'] for m in recent_metrics),
                'max': max(m['totals']['overall_success_rate'] for m in recent_metrics),
                'avg': sum(m['totals']['overall_success_rate'] for m in recent_metrics) / len(recent_metrics)
            },
            'instance_health': {
                'avg_healthy': sum(m['totals']['healthy_instances'] for m in recent_metrics) / len(recent_metrics),
                'avg_degraded': sum(m['totals']['degraded_instances'] for m in recent_metrics) / len(recent_metrics),
                'avg_unhealthy': sum(m['totals']['unhealthy_instances'] for m in recent_metrics) / len(recent_metrics)
            }
        }
        
        return summary


class QueryOptimizer:
    """AI-powered query optimization system"""
    
    def __init__(self):
        self.query_cache: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []
    
    def optimize_query(self, query: str, db_type: DatabaseType, context: Optional[Dict[str, Any]] = None) -> str:
        """Optimize query using AI and historical performance data"""
        # Simple query optimization - can be enhanced with ML models
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash in self.query_cache:
            cached_optimization = self.query_cache[query_hash]
            if cached_optimization['performance_score'] > 0.8:
                return cached_optimization['optimized_query']
        
        # Apply database-specific optimizations
        optimized_query = query
        
        if db_type in [DatabaseType.POSTGRESQL, DatabaseType.TIMESCALEDB]:
            optimized_query = self._optimize_postgresql_query(query, context)
        elif db_type == DatabaseType.MONGODB:
            optimized_query = self._optimize_mongodb_query(query, context)
        elif db_type == DatabaseType.CLICKHOUSE:
            optimized_query = self._optimize_clickhouse_query(query, context)
        elif db_type == DatabaseType.ELASTICSEARCH:
            optimized_query = self._optimize_elasticsearch_query(query, context)
        
        # Cache optimization
        self.query_cache[query_hash] = {
            'original_query': query,
            'optimized_query': optimized_query,
            'performance_score': 0.7,  # Default score, should be measured
            'optimization_time': datetime.now().isoformat()
        }
        
        return optimized_query
    
    def _optimize_postgresql_query(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """PostgreSQL specific query optimizations"""
        optimized = query
        
        # Add basic optimizations
        if 'SELECT *' in query and context and 'required_fields' in context:
            # Replace SELECT * with specific fields
            fields = ', '.join(context['required_fields'])
            optimized = query.replace('SELECT *', f'SELECT {fields}')
        
        # Add LIMIT if not present for large result sets
        if 'LIMIT' not in query.upper() and 'SELECT' in query.upper():
            optimized += ' LIMIT 1000'
        
        return optimized
    
    def _optimize_mongodb_query(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """MongoDB specific query optimizations"""
        # MongoDB query optimization logic
        return query
    
    def _optimize_clickhouse_query(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """ClickHouse specific query optimizations"""
        optimized = query
        
        # Add SAMPLE clause for large analytics queries
        if 'GROUP BY' in query.upper() and 'SAMPLE' not in query.upper():
            # Add sampling for large aggregations
            optimized = query.replace('FROM ', 'FROM (SELECT * FROM ')
            optimized += ' SAMPLE 0.1) '
        
        return optimized
    
    def _optimize_elasticsearch_query(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Elasticsearch specific query optimizations"""
        # Elasticsearch query optimization logic
        return query


class CacheManager:
    """Intelligent caching system for database results"""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.cache_stats: Dict[str, int] = {'hits': 0, 'misses': 0}
        self.max_cache_size = 1000
        self.default_ttl = 300  # 5 minutes
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result"""
        if key in self.cache:
            cache_entry = self.cache[key]
            if cache_entry['expires_at'] > datetime.now():
                self.cache_stats['hits'] += 1
                return cache_entry['data']
            else:
                del self.cache[key]
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Set cached result"""
        if len(self.cache) >= self.max_cache_size:
            # Simple LRU eviction - remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['created_at'])
            del self.cache[oldest_key]
        
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            'data': data,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=ttl),
            'access_count': 0
        }
    
    def generate_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate cache key from query and parameters"""
        combined = f"{query}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_entries': len(self.cache),
            'max_size': self.max_cache_size,
            'hit_rate_percent': hit_rate,
            'total_hits': self.cache_stats['hits'],
            'total_misses': self.cache_stats['misses']
        }
