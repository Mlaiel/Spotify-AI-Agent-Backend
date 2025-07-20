"""
🚀 Isolation Engine - Moteur Central d'Isolation des Données
===========================================================

Moteur ultra-avancé d'isolation des données multi-tenant avec 
performance enterprise-grade et sécurité paranoid-level.

Author: Architecte Microservices - Fahed Mlaiel
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from .tenant_context import TenantContext, IsolationLevel
from .data_partition import DataPartition, PartitionConfig, PartitionType, PartitionStrategy
from ..strategies.database_level import DatabaseLevelStrategy
from ..strategies.schema_level import SchemaLevelStrategy
from ..strategies.row_level import RowLevelStrategy
from ..strategies.hybrid_strategy import HybridStrategy
from ..managers.connection_manager import ConnectionManager
from ..managers.cache_manager import CacheManager
from ..managers.security_manager import SecurityManager
from ..monitoring.isolation_monitor import IsolationMonitor
from ..exceptions import IsolationLevelError, DataIsolationError


class EngineState(Enum):
    """États du moteur d'isolation"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    STOPPED = "stopped"
    ERROR = "error"


class PerformanceMode(Enum):
    """Modes de performance"""
    CONSERVATIVE = "conservative"  # Privilégie la sécurité
    BALANCED = "balanced"         # Équilibre perf/sécurité
    AGGRESSIVE = "aggressive"     # Privilégie la performance


@dataclass
class EngineConfig:
    """Configuration du moteur d'isolation"""
    isolation_level: IsolationLevel = IsolationLevel.STRICT
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    
    # Partition configuration
    partition_config: Optional[PartitionConfig] = None
    auto_partition: bool = True
    
    # Connection management
    max_connections_per_tenant: int = 10
    connection_timeout: int = 30
    connection_retry_attempts: int = 3
    
    # Cache configuration
    cache_enabled: bool = True
    cache_ttl: int = 3600
    cache_size_limit: int = 1000
    
    # Security configuration
    encryption_enabled: bool = True
    audit_enabled: bool = True
    monitoring_enabled: bool = True
    
    # Performance configuration
    query_timeout: int = 30
    bulk_operation_size: int = 1000
    parallel_operations: int = 4
    
    # Maintenance configuration
    auto_maintenance: bool = True
    maintenance_interval: int = 3600  # seconds
    health_check_interval: int = 60   # seconds


@dataclass
class EngineMetrics:
    """Métriques du moteur d'isolation"""
    requests_total: int = 0
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    active_tenants: int = 0
    active_connections: int = 0
    cache_hit_rate: float = 0.0
    memory_usage: int = 0
    cpu_usage: float = 0.0
    
    # Security metrics
    security_violations: int = 0
    failed_authentications: int = 0
    
    # Performance metrics
    slow_queries: int = 0
    deadlocks: int = 0
    timeouts: int = 0


class IsolationStrategy:
    """Interface pour les stratégies d'isolation"""
    
    async def initialize(self, config: EngineConfig):
        """Initialise la stratégie"""
        pass
    
    async def apply_isolation(self, context: TenantContext, query: Any) -> Any:
        """Applique l'isolation à une requête"""
        pass
    
    async def validate_access(self, context: TenantContext, resource: str) -> bool:
        """Valide l'accès à une ressource"""
        pass
    
    async def cleanup(self):
        """Nettoie les ressources"""
        pass


class IsolationEngine:
    """
    Moteur central d'isolation des données multi-tenant
    
    Features:
    - Support de multiples stratégies d'isolation
    - Performance monitoring en temps réel
    - Auto-scaling et load balancing
    - Security monitoring et audit
    - Health checks automatiques
    - Connection pooling intelligent
    - Cache multi-niveau
    - Recovery automatique
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.state = EngineState.INITIALIZING
        self.logger = logging.getLogger("isolation.engine")
        
        # Core components
        self.connection_manager = ConnectionManager(config)
        self.cache_manager = CacheManager(config) if config.cache_enabled else None
        self.security_manager = SecurityManager(config)
        self.isolation_monitor = IsolationMonitor(config) if config.monitoring_enabled else None
        
        # Strategies
        self.strategies: Dict[IsolationLevel, IsolationStrategy] = {}
        
        # Data partitioning
        self.data_partition: Optional[DataPartition] = None
        
        # Metrics and monitoring
        self.metrics = EngineMetrics()
        self._metrics_lock = threading.Lock()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._executor = ThreadPoolExecutor(max_workers=config.parallel_operations)
        
        # Request tracking
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._request_history: List[Dict[str, Any]] = []
        
        # Initialize components
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialise le moteur d'isolation"""
        try:
            self.logger.info("Initializing isolation engine...")
            
            # Initialize strategies
            await self._initialize_strategies()
            
            # Initialize data partitioning
            if self.config.auto_partition:
                await self._initialize_partitioning()
            
            # Initialize monitoring
            if self.isolation_monitor:
                await self.isolation_monitor.start()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.state = EngineState.RUNNING
            self.logger.info("Isolation engine initialized successfully")
            
        except Exception as e:
            self.state = EngineState.ERROR
            self.logger.error(f"Failed to initialize isolation engine: {e}")
            raise DataIsolationError(f"Engine initialization failed: {e}")
    
    async def _initialize_strategies(self):
        """Initialise les stratégies d'isolation"""
        strategy_classes = {
            IsolationLevel.NONE: None,  # No isolation
            IsolationLevel.BASIC: RowLevelStrategy,
            IsolationLevel.STRICT: SchemaLevelStrategy,
            IsolationLevel.PARANOID: HybridStrategy
        }
        
        for level, strategy_class in strategy_classes.items():
            if strategy_class:
                strategy = strategy_class()
                await strategy.initialize(self.config)
                self.strategies[level] = strategy
                self.logger.debug(f"Initialized strategy for level: {level.value}")
    
    async def _initialize_partitioning(self):
        """Initialise le partitionnement des données"""
        if not self.config.partition_config:
            # Create default partition config
            self.config.partition_config = PartitionConfig(
                partition_type=PartitionType.HORIZONTAL,
                strategy=PartitionStrategy.CONSISTENT_HASH,
                partition_count=16,
                auto_scaling=True,
                monitoring_enabled=True
            )
        
        self.data_partition = DataPartition(self.config.partition_config)
        self.logger.info("Data partitioning initialized")
    
    async def _start_background_tasks(self):
        """Démarre les tâches en arrière-plan"""
        if self.config.auto_maintenance:
            self._background_tasks.append(
                asyncio.create_task(self._maintenance_loop())
            )
        
        self._background_tasks.append(
            asyncio.create_task(self._health_check_loop())
        )
        
        self._background_tasks.append(
            asyncio.create_task(self._metrics_collection_loop())
        )
        
        self.logger.debug(f"Started {len(self._background_tasks)} background tasks")
    
    async def _maintenance_loop(self):
        """Boucle de maintenance automatique"""
        while self.state in [EngineState.RUNNING, EngineState.DEGRADED]:
            try:
                await self._perform_maintenance()
                await asyncio.sleep(self.config.maintenance_interval)
            except Exception as e:
                self.logger.error(f"Maintenance error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def _health_check_loop(self):
        """Boucle de vérification de santé"""
        while self.state != EngineState.STOPPED:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_collection_loop(self):
        """Boucle de collecte des métriques"""
        while self.state != EngineState.STOPPED:
            try:
                await self._collect_metrics()
                await asyncio.sleep(10)  # Collect every 10 seconds
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _perform_maintenance(self):
        """Effectue la maintenance automatique"""
        self.logger.debug("Performing automated maintenance...")
        
        # Connection pool maintenance
        await self.connection_manager.cleanup_idle_connections()
        
        # Cache maintenance
        if self.cache_manager:
            await self.cache_manager.cleanup_expired()
        
        # Partition maintenance
        if self.data_partition:
            await self.data_partition.balance_partitions()
        
        # Metrics cleanup
        self._cleanup_old_metrics()
        
        self.logger.debug("Maintenance completed")
    
    async def _perform_health_check(self):
        """Effectue une vérification de santé"""
        health_issues = []
        
        # Check connection manager health
        if not await self.connection_manager.health_check():
            health_issues.append("Connection manager unhealthy")
        
        # Check cache manager health
        if self.cache_manager and not await self.cache_manager.health_check():
            health_issues.append("Cache manager unhealthy")
        
        # Check strategy health
        for level, strategy in self.strategies.items():
            if hasattr(strategy, 'health_check') and not await strategy.health_check():
                health_issues.append(f"Strategy {level.value} unhealthy")
        
        # Update engine state based on health
        if health_issues:
            if self.state == EngineState.RUNNING:
                self.state = EngineState.DEGRADED
                self.logger.warning(f"Engine degraded. Issues: {health_issues}")
        else:
            if self.state == EngineState.DEGRADED:
                self.state = EngineState.RUNNING
                self.logger.info("Engine health restored")
    
    async def _collect_metrics(self):
        """Collecte les métriques système"""
        with self._metrics_lock:
            # Update basic metrics
            self.metrics.active_tenants = len(self._active_requests)
            self.metrics.active_connections = await self.connection_manager.get_active_count()
            
            if self.cache_manager:
                self.metrics.cache_hit_rate = await self.cache_manager.get_hit_rate()
            
            # Calculate rates and averages
            recent_requests = self._request_history[-100:]  # Last 100 requests
            if recent_requests:
                response_times = [r['response_time'] for r in recent_requests if 'response_time' in r]
                if response_times:
                    self.metrics.average_response_time = sum(response_times) / len(response_times)
    
    def _cleanup_old_metrics(self):
        """Nettoie les anciennes métriques"""
        # Keep only last 1000 requests in history
        if len(self._request_history) > 1000:
            self._request_history = self._request_history[-1000:]
    
    async def apply_isolation(
        self, 
        context: TenantContext, 
        operation: str,
        target: Any,
        **kwargs
    ) -> Any:
        """
        Applique l'isolation des données pour une opération
        
        Args:
            context: Contexte du tenant
            operation: Type d'opération (query, insert, update, delete)
            target: Cible de l'opération (table, collection, etc.)
            **kwargs: Arguments supplémentaires
        
        Returns:
            Résultat de l'opération avec isolation appliquée
        """
        request_id = f"{context.tenant_id}_{datetime.now().timestamp()}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Track request
            self._active_requests[request_id] = {
                "tenant_id": context.tenant_id,
                "operation": operation,
                "start_time": start_time,
                "target": str(target)
            }
            
            # Security validation
            if not await self.security_manager.validate_access(context, operation, target):
                raise DataIsolationError(f"Access denied for tenant {context.tenant_id}")
            
            # Get appropriate strategy
            strategy = self.strategies.get(context.isolation_level)
            if not strategy:
                raise IsolationLevelError(f"No strategy for isolation level: {context.isolation_level}")
            
            # Apply isolation
            result = await strategy.apply_isolation(context, target, **kwargs)
            
            # Log success
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._record_request_success(request_id, response_time)
            
            return result
            
        except Exception as e:
            # Log error
            self._record_request_error(request_id, str(e))
            self.logger.error(f"Isolation failed for tenant {context.tenant_id}: {e}")
            raise
        
        finally:
            # Cleanup request tracking
            self._active_requests.pop(request_id, None)
    
    def _record_request_success(self, request_id: str, response_time: float):
        """Enregistre une requête réussie"""
        request_info = self._active_requests.get(request_id, {})
        request_info.update({
            "status": "success",
            "response_time": response_time,
            "completed_at": datetime.now(timezone.utc)
        })
        
        self._request_history.append(request_info)
        
        with self._metrics_lock:
            self.metrics.requests_total += 1
    
    def _record_request_error(self, request_id: str, error: str):
        """Enregistre une requête en erreur"""
        request_info = self._active_requests.get(request_id, {})
        request_info.update({
            "status": "error",
            "error": error,
            "completed_at": datetime.now(timezone.utc)
        })
        
        self._request_history.append(request_info)
        
        with self._metrics_lock:
            self.metrics.requests_total += 1
    
    async def get_tenant_connection(self, context: TenantContext):
        """Obtient une connexion pour un tenant"""
        return await self.connection_manager.get_connection(context)
    
    @asynccontextmanager
    async def tenant_transaction(self, context: TenantContext):
        """Context manager pour les transactions tenant"""
        connection = await self.get_tenant_connection(context)
        transaction = await connection.begin()
        
        try:
            yield connection
            await transaction.commit()
        except Exception:
            await transaction.rollback()
            raise
        finally:
            await self.connection_manager.return_connection(context, connection)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du moteur"""
        with self._metrics_lock:
            metrics_dict = {
                "state": self.state.value,
                "config": {
                    "isolation_level": self.config.isolation_level.value,
                    "performance_mode": self.config.performance_mode.value,
                    "cache_enabled": self.config.cache_enabled,
                    "monitoring_enabled": self.config.monitoring_enabled
                },
                "metrics": {
                    "requests_total": self.metrics.requests_total,
                    "requests_per_second": self.metrics.requests_per_second,
                    "average_response_time": self.metrics.average_response_time,
                    "error_rate": self.metrics.error_rate,
                    "active_tenants": self.metrics.active_tenants,
                    "active_connections": self.metrics.active_connections,
                    "cache_hit_rate": self.metrics.cache_hit_rate
                },
                "background_tasks": len(self._background_tasks),
                "active_requests": len(self._active_requests)
            }
        
        # Add partition stats if available
        if self.data_partition:
            metrics_dict["partition_stats"] = await self.data_partition.get_partition_stats()
        
        return metrics_dict
    
    async def shutdown(self):
        """Arrête proprement le moteur d'isolation"""
        self.logger.info("Shutting down isolation engine...")
        self.state = EngineState.STOPPED
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Cleanup strategies
        for strategy in self.strategies.values():
            await strategy.cleanup()
        
        # Cleanup managers
        await self.connection_manager.cleanup()
        if self.cache_manager:
            await self.cache_manager.cleanup()
        await self.security_manager.cleanup()
        
        # Cleanup partition manager
        if self.data_partition:
            await self.data_partition.cleanup()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        self.logger.info("Isolation engine shutdown completed")


# Factory function for creating isolation engines
def create_isolation_engine(
    isolation_level: IsolationLevel = IsolationLevel.STRICT,
    performance_mode: PerformanceMode = PerformanceMode.BALANCED,
    **kwargs
) -> IsolationEngine:
    """Factory pour créer un moteur d'isolation"""
    config = EngineConfig(
        isolation_level=isolation_level,
        performance_mode=performance_mode,
        **kwargs
    )
    return IsolationEngine(config)


# Global engine instance
_global_engine: Optional[IsolationEngine] = None


def get_isolation_engine() -> IsolationEngine:
    """Retourne l'instance globale du moteur d'isolation"""
    global _global_engine
    if _global_engine is None:
        _global_engine = create_isolation_engine()
    return _global_engine


def set_isolation_engine(engine: IsolationEngine):
    """Définit l'instance globale du moteur d'isolation"""
    global _global_engine
    _global_engine = engine
