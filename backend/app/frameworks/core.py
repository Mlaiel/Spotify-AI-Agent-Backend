"""
🚀 FRAMEWORK CORE - ORCHESTRATEUR CENTRAL ENTERPRISE
Expert Team: Lead Developer + AI Architect, Microservices Architect

Module central d'orchestration des frameworks avec architecture ultra-avancée industrielle
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import time
import weakref
from functools import wraps

# Monitoring et observabilité
import prometheus_client
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Circuit breaker et resilience
from circuitbreaker import circuit
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration avancée
from pydantic import BaseSettings, Field
from typing_extensions import Annotated


class FrameworkStatus(Enum):
    """États des frameworks"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class FrameworkHealth:
    """Santé d'un framework"""
    status: FrameworkStatus
    last_check: float
    error_count: int = 0
    latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FrameworkConfig(BaseSettings):
    """Configuration globale des frameworks"""
    
    # Core settings
    max_workers: int = Field(default=10, description="Maximum worker threads")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    startup_timeout: int = Field(default=120, description="Framework startup timeout")
    shutdown_timeout: int = Field(default=60, description="Framework shutdown timeout")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    jaeger_endpoint: str = Field(default="http://localhost:14268/api/traces")
    
    # Resilience
    circuit_breaker_enabled: bool = Field(default=True)
    max_failures: int = Field(default=5, description="Circuit breaker failure threshold")
    recovery_timeout: int = Field(default=60, description="Circuit breaker recovery timeout")
    
    # Performance
    async_pool_size: int = Field(default=50, description="Async connection pool size")
    cache_ttl: int = Field(default=300, description="Default cache TTL in seconds")
    
    class Config:
        env_prefix = "FRAMEWORK_"


class BaseFramework(ABC):
    """Interface de base pour tous les frameworks"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.status = FrameworkStatus.UNINITIALIZED
        self.health = FrameworkHealth(
            status=self.status,
            last_check=time.time()
        )
        self.logger = logging.getLogger(f"framework.{name}")
        self._startup_time: Optional[float] = None
        self._shutdown_time: Optional[float] = None
        
        # Métriques Prometheus
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialise les métriques Prometheus"""
        self.startup_counter = prometheus_client.Counter(
            f'framework_{self.name}_startup_total',
            'Framework startup attempts'
        )
        self.health_gauge = prometheus_client.Gauge(
            f'framework_{self.name}_health',
            'Framework health status (1=healthy, 0=unhealthy)'
        )
        self.latency_histogram = prometheus_client.Histogram(
            f'framework_{self.name}_operation_duration_seconds',
            'Framework operation duration'
        )
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialise le framework"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Arrête le framework"""
        pass
    
    @abstractmethod
    async def health_check(self) -> FrameworkHealth:
        """Vérifie la santé du framework"""
        pass
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques du framework"""
        return {
            "name": self.name,
            "status": self.status.value,
            "startup_time": self._startup_time,
            "uptime": time.time() - (self._startup_time or time.time()),
            "health": self.health.__dict__
        }
    
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def safe_operation(self, operation: Callable, *args, **kwargs):
        """Exécute une opération avec circuit breaker"""
        try:
            with self.latency_histogram.time():
                result = await operation(*args, **kwargs)
            return result
        except Exception as e:
            self.logger.error(f"Operation failed: {e}")
            raise


class FrameworkOrchestrator:
    """
    🎯 ORCHESTRATEUR CENTRAL DES FRAMEWORKS
    
    Architecture enterprise avec:
    - Gestion du cycle de vie
    - Health monitoring avancé
    - Circuit breakers
    - Métriques et observabilité
    - Resilience patterns
    """
    
    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config or FrameworkConfig()
        self.frameworks: Dict[str, BaseFramework] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.logger = logging.getLogger("framework.orchestrator")
        
        # Monitoring et observabilité
        self._init_monitoring()
        
        # Health checking
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Métriques globales
        self.frameworks_gauge = prometheus_client.Gauge(
            'frameworks_total',
            'Total number of registered frameworks',
            ['status']
        )
        self.orchestrator_operations = prometheus_client.Counter(
            'orchestrator_operations_total',
            'Total orchestrator operations',
            ['operation', 'result']
        )
    
    def _init_monitoring(self):
        """Initialise le monitoring et tracing"""
        if self.config.enable_tracing:
            trace.set_tracer_provider(TracerProvider())
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
        self.tracer = trace.get_tracer(__name__)
    
    def register_framework(
        self, 
        framework: BaseFramework, 
        dependencies: Optional[List[str]] = None
    ):
        """
        Enregistre un framework avec ses dépendances
        
        Args:
            framework: Instance du framework
            dependencies: Liste des frameworks requis
        """
        with self.tracer.start_as_current_span("register_framework") as span:
            span.set_attribute("framework.name", framework.name)
            
            if framework.name in self.frameworks:
                raise ValueError(f"Framework {framework.name} already registered")
            
            self.frameworks[framework.name] = framework
            self.dependencies[framework.name] = dependencies or []
            
            self.logger.info(f"Registered framework: {framework.name}")
            self.orchestrator_operations.labels(
                operation="register", 
                result="success"
            ).inc()
    
    async def initialize_framework(self, name: str) -> bool:
        """
        Initialise un framework spécifique avec ses dépendances
        
        Args:
            name: Nom du framework
            
        Returns:
            bool: Succès de l'initialisation
        """
        with self.tracer.start_as_current_span("initialize_framework") as span:
            span.set_attribute("framework.name", name)
            
            if name not in self.frameworks:
                raise ValueError(f"Framework {name} not registered")
            
            framework = self.frameworks[name]
            
            if framework.status == FrameworkStatus.RUNNING:
                self.logger.info(f"Framework {name} already running")
                return True
            
            # Vérifier et initialiser les dépendances
            for dep in self.dependencies[name]:
                if not await self.initialize_framework(dep):
                    self.logger.error(f"Failed to initialize dependency {dep} for {name}")
                    return False
            
            # Initialiser le framework
            try:
                framework.status = FrameworkStatus.INITIALIZING
                framework._startup_time = time.time()
                framework.startup_counter.inc()
                
                self.logger.info(f"Initializing framework: {name}")
                
                success = await asyncio.wait_for(
                    framework.initialize(),
                    timeout=self.config.startup_timeout
                )
                
                if success:
                    framework.status = FrameworkStatus.RUNNING
                    framework.health_gauge.set(1)
                    self.frameworks_gauge.labels(status="running").inc()
                    self.logger.info(f"Framework {name} initialized successfully")
                    self.orchestrator_operations.labels(
                        operation="initialize", 
                        result="success"
                    ).inc()
                else:
                    framework.status = FrameworkStatus.FAILED
                    framework.health_gauge.set(0)
                    self.frameworks_gauge.labels(status="failed").inc()
                    self.orchestrator_operations.labels(
                        operation="initialize", 
                        result="failure"
                    ).inc()
                
                return success
                
            except asyncio.TimeoutError:
                framework.status = FrameworkStatus.FAILED
                self.logger.error(f"Framework {name} initialization timeout")
                self.orchestrator_operations.labels(
                    operation="initialize", 
                    result="timeout"
                ).inc()
                return False
            except Exception as e:
                framework.status = FrameworkStatus.FAILED
                self.logger.error(f"Framework {name} initialization failed: {e}")
                self.orchestrator_operations.labels(
                    operation="initialize", 
                    result="error"
                ).inc()
                return False
    
    async def initialize_all_frameworks(self) -> Dict[str, bool]:
        """
        Initialise tous les frameworks dans l'ordre des dépendances
        
        Returns:
            Dict[str, bool]: Résultats d'initialisation par framework
        """
        with self.tracer.start_as_current_span("initialize_all_frameworks"):
            results = {}
            
            # Tri topologique des frameworks
            ordered_frameworks = self._topological_sort()
            
            for name in ordered_frameworks:
                results[name] = await self.initialize_framework(name)
            
            # Démarrer le health checking
            if not self._health_check_task:
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )
            
            return results
    
    def _topological_sort(self) -> List[str]:
        """Tri topologique des frameworks selon leurs dépendances"""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            for dep in self.dependencies[name]:
                visit(dep)
            temp_visited.remove(name)
            visited.add(name)
            result.append(name)
        
        for name in self.frameworks:
            if name not in visited:
                visit(name)
        
        return result
    
    async def _health_check_loop(self):
        """Boucle de vérification de santé des frameworks"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                for name, framework in self.frameworks.items():
                    if framework.status == FrameworkStatus.RUNNING:
                        try:
                            health = await framework.health_check()
                            framework.health = health
                            
                            if health.status == FrameworkStatus.DEGRADED:
                                self.logger.warning(f"Framework {name} is degraded")
                                framework.health_gauge.set(0.5)
                            elif health.status == FrameworkStatus.FAILED:
                                self.logger.error(f"Framework {name} failed health check")
                                framework.health_gauge.set(0)
                                framework.status = FrameworkStatus.FAILED
                            else:
                                framework.health_gauge.set(1)
                                
                        except Exception as e:
                            self.logger.error(f"Health check failed for {name}: {e}")
                            framework.health.error_count += 1
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
    
    async def get_health_status(self) -> Dict[str, FrameworkHealth]:
        """Récupère le statut de santé de tous les frameworks"""
        return {name: fw.health for name, fw in self.frameworks.items()}
    
    async def shutdown_framework(self, name: str) -> bool:
        """Arrête un framework spécifique"""
        with self.tracer.start_as_current_span("shutdown_framework") as span:
            span.set_attribute("framework.name", name)
            
            if name not in self.frameworks:
                return False
            
            framework = self.frameworks[name]
            
            if framework.status in [FrameworkStatus.SHUTDOWN, FrameworkStatus.SHUTTING_DOWN]:
                return True
            
            try:
                framework.status = FrameworkStatus.SHUTTING_DOWN
                framework._shutdown_time = time.time()
                
                success = await asyncio.wait_for(
                    framework.shutdown(),
                    timeout=self.config.shutdown_timeout
                )
                
                if success:
                    framework.status = FrameworkStatus.SHUTDOWN
                    framework.health_gauge.set(0)
                    self.frameworks_gauge.labels(status="shutdown").inc()
                
                return success
                
            except Exception as e:
                self.logger.error(f"Framework {name} shutdown failed: {e}")
                return False
    
    async def shutdown_all(self) -> Dict[str, bool]:
        """Arrête tous les frameworks dans l'ordre inverse"""
        with self.tracer.start_as_current_span("shutdown_all_frameworks"):
            # Arrêter le health checking
            self._shutdown_event.set()
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            results = {}
            ordered_frameworks = list(reversed(self._topological_sort()))
            
            for name in ordered_frameworks:
                results[name] = await self.shutdown_framework(name)
            
            # Fermer l'executor
            self.executor.shutdown(wait=True)
            
            return results
    
    def get_framework(self, name: str) -> Optional[BaseFramework]:
        """Récupère une instance de framework"""
        return self.frameworks.get(name)
    
    async def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques de l'orchestrateur"""
        framework_statuses = {}
        for status in FrameworkStatus:
            count = sum(1 for fw in self.frameworks.values() if fw.status == status)
            framework_statuses[status.value] = count
        
        return {
            "total_frameworks": len(self.frameworks),
            "status_distribution": framework_statuses,
            "dependencies": self.dependencies,
            "config": self.config.dict(),
            "uptime": time.time() - (self.frameworks and min(
                fw._startup_time for fw in self.frameworks.values() 
                if fw._startup_time
            ) or time.time())
        }


# Instance globale de l'orchestrateur
framework_orchestrator = FrameworkOrchestrator()


# Décorateurs utilitaires
def framework_dependency(dependency_name: str):
    """Décorateur pour marquer les dépendances de framework"""
    def decorator(cls):
        if not hasattr(cls, '_framework_dependencies'):
            cls._framework_dependencies = []
        cls._framework_dependencies.append(dependency_name)
        return cls
    return decorator


@asynccontextmanager
async def framework_context(orchestrator: FrameworkOrchestrator):
    """Context manager pour l'orchestrateur de frameworks"""
    try:
        await orchestrator.initialize_all_frameworks()
        yield orchestrator
    finally:
        await orchestrator.shutdown_all()


# Export des classes principales
__all__ = [
    'FrameworkOrchestrator',
    'BaseFramework', 
    'FrameworkStatus',
    'FrameworkHealth',
    'FrameworkConfig',
    'framework_orchestrator',
    'framework_dependency',
    'framework_context'
]
