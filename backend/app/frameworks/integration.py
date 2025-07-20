# 🎵 Spotify AI Agent - Framework Integration
# ==========================================
# 
# Système d'intégration enterprise pour tous les frameworks
# Integration layer pour microservices, ML, monitoring et sécurité
#
# 🎖️ Expert: Architecte Enterprise + Lead Dev + DevOps
#
# 👨‍💻 Développé par: Fahed Mlaiel
# ==========================================

"""
🔗 Enterprise Framework Integration System
==========================================

Comprehensive integration layer providing:
- Framework orchestration and lifecycle management
- Service discovery and registration
- Inter-framework communication and messaging
- Configuration management across frameworks
- Health monitoring and fault tolerance
- Event-driven architecture and workflow orchestration
- Plugin system and extensibility
"""

import asyncio
import logging
import json
import threading
import weakref
from typing import Dict, Any, List, Optional, Callable, Type, Union, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod
import inspect
from pathlib import Path
import importlib
import sys

# Core dependencies
from fastapi import FastAPI, BackgroundTasks
from dependency_injector import containers, providers
from pydantic import BaseModel, Field, validator
import aioredis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge
import schedule

# Internal framework imports
from .core import CoreFramework
from .hybrid_backend import HybridBackendFramework
from .microservices import MicroservicesFramework
from .ml_frameworks import MLFrameworksManager
from .monitoring import MonitoringFramework
from .security import SecurityFramework

logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Statuts d'intégration des frameworks"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


class FrameworkType(Enum):
    """Types de frameworks supportés"""
    CORE = "core"
    BACKEND = "backend"
    MICROSERVICES = "microservices"
    ML = "ml"
    MONITORING = "monitoring"
    SECURITY = "security"
    CUSTOM = "custom"


class MessageType(Enum):
    """Types de messages inter-frameworks"""
    COMMAND = "command"
    EVENT = "event"
    QUERY = "query"
    RESPONSE = "response"
    NOTIFICATION = "notification"


@dataclass
class FrameworkInfo:
    """Informations sur un framework"""
    name: str
    type: FrameworkType
    version: str
    instance: Any
    status: IntegrationStatus = IntegrationStatus.INITIALIZING
    dependencies: List[str] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    startup_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None


@dataclass
class IntegrationMessage:
    """Message d'intégration entre frameworks"""
    id: str
    type: MessageType
    source: str
    target: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    timeout: Optional[int] = None
    retry_count: int = 0


class FrameworkInterface(Protocol):
    """Interface commune pour tous les frameworks"""
    
    @property
    def name(self) -> str:
        """Nom du framework"""
        ...
    
    @property
    def version(self) -> str:
        """Version du framework"""
        ...
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialisation du framework"""
        ...
    
    async def start(self) -> bool:
        """Démarrage du framework"""
        ...
    
    async def stop(self) -> bool:
        """Arrêt du framework"""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Contrôle de santé"""
        ...
    
    async def handle_message(self, message: IntegrationMessage) -> Optional[IntegrationMessage]:
        """Gestion des messages inter-frameworks"""
        ...


class EventBus:
    """Bus d'événements pour communication inter-frameworks"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[IntegrationMessage] = []
        self.max_history = 1000
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """S'abonner à un type d'événement"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
        logger.debug(f"Subscribed to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Se désabonner d'un type d'événement"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)
            logger.debug(f"Unsubscribed from event type: {event_type}")
    
    async def publish(self, message: IntegrationMessage) -> None:
        """Publier un message/événement"""
        # Ajouter à l'historique
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        # Notifier les abonnés
        event_type = f"{message.source}.{message.type.value}"
        if event_type in self.subscribers:
            tasks = []
            for callback in self.subscribers[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback(message))
                else:
                    # Exécuter les callbacks synchrones dans un thread
                    tasks.append(asyncio.get_event_loop().run_in_executor(
                        None, callback, message
                    ))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug(f"Published message: {message.id} from {message.source} to {message.target}")
    
    def get_message_history(self, 
                           source: Optional[str] = None,
                           target: Optional[str] = None,
                           message_type: Optional[MessageType] = None,
                           limit: int = 100) -> List[IntegrationMessage]:
        """Récupérer l'historique des messages"""
        filtered_messages = self.message_history
        
        if source:
            filtered_messages = [m for m in filtered_messages if m.source == source]
        if target:
            filtered_messages = [m for m in filtered_messages if m.target == target]
        if message_type:
            filtered_messages = [m for m in filtered_messages if m.type == message_type]
        
        return filtered_messages[-limit:]


class ServiceRegistry:
    """Registre de services pour découverte automatique"""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, datetime] = {}
    
    def register_service(self, 
                        service_name: str,
                        endpoint: str,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Enregistrer un service"""
        self.services[service_name] = {
            "endpoint": endpoint,
            "metadata": metadata or {},
            "registered_at": datetime.utcnow()
        }
        
        logger.info(f"Registered service: {service_name} at {endpoint}")
        return True
    
    def unregister_service(self, service_name: str) -> bool:
        """Désenregistrer un service"""
        if service_name in self.services:
            del self.services[service_name]
            if service_name in self.health_checks:
                del self.health_checks[service_name]
            
            logger.info(f"Unregistered service: {service_name}")
            return True
        return False
    
    def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Découvrir un service"""
        return self.services.get(service_name)
    
    def list_services(self, filter_healthy: bool = True) -> Dict[str, Dict[str, Any]]:
        """Lister tous les services"""
        if not filter_healthy:
            return self.services.copy()
        
        # Filtrer les services sains
        healthy_services = {}
        for name, info in self.services.items():
            last_check = self.health_checks.get(name)
            if last_check and (datetime.utcnow() - last_check).seconds < 60:
                healthy_services[name] = info
        
        return healthy_services
    
    def update_health_status(self, service_name: str, is_healthy: bool) -> None:
        """Mettre à jour le statut de santé d'un service"""
        if is_healthy:
            self.health_checks[service_name] = datetime.utcnow()
        elif service_name in self.health_checks:
            del self.health_checks[service_name]


class WorkflowEngine:
    """Moteur de workflow pour orchestration de tâches"""
    
    def __init__(self, integration_manager):
        self.integration_manager = integration_manager
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
    
    def register_workflow(self, 
                         workflow_id: str,
                         steps: List[Dict[str, Any]],
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Enregistrer un workflow"""
        self.workflows[workflow_id] = {
            "steps": steps,
            "metadata": metadata or {},
            "created_at": datetime.utcnow()
        }
        
        logger.info(f"Registered workflow: {workflow_id}")
        return True
    
    async def execute_workflow(self, 
                              workflow_id: str,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Exécuter un workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        execution_context = context or {}
        
        logger.info(f"Starting workflow execution: {workflow_id}")
        
        results = []
        for i, step in enumerate(workflow["steps"]):
            try:
                step_result = await self._execute_step(step, execution_context)
                results.append(step_result)
                
                # Mettre à jour le contexte avec les résultats
                execution_context.update(step_result.get("outputs", {}))
                
            except Exception as e:
                logger.error(f"Workflow step {i} failed: {e}")
                return {
                    "status": "failed",
                    "error": str(e),
                    "completed_steps": i,
                    "results": results
                }
        
        logger.info(f"Workflow completed successfully: {workflow_id}")
        return {
            "status": "completed",
            "results": results,
            "execution_context": execution_context
        }
    
    async def _execute_step(self, 
                           step: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Exécuter une étape de workflow"""
        step_type = step.get("type")
        
        if step_type == "framework_call":
            # Appel à un framework spécifique
            framework_name = step["framework"]
            method = step["method"]
            params = step.get("params", {})
            
            # Substitution des variables de contexte
            for key, value in params.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    var_name = value[2:-1]
                    params[key] = context.get(var_name, value)
            
            framework = self.integration_manager.get_framework(framework_name)
            if not framework:
                raise ValueError(f"Framework not found: {framework_name}")
            
            result = await getattr(framework.instance, method)(**params)
            return {"outputs": {"result": result}}
        
        elif step_type == "condition":
            # Évaluation conditionnelle
            condition = step["condition"]
            # Évaluation simple basée sur le contexte
            if self._evaluate_condition(condition, context):
                if "then_steps" in step:
                    return await self._execute_substeps(step["then_steps"], context)
            else:
                if "else_steps" in step:
                    return await self._execute_substeps(step["else_steps"], context)
            
            return {"outputs": {}}
        
        elif step_type == "parallel":
            # Exécution parallèle
            tasks = []
            for substep in step["steps"]:
                tasks.append(self._execute_step(substep, context))
            
            results = await asyncio.gather(*tasks)
            return {"outputs": {"parallel_results": results}}
        
        elif step_type == "delay":
            # Délai d'attente
            delay_seconds = step.get("seconds", 1)
            await asyncio.sleep(delay_seconds)
            return {"outputs": {}}
        
        else:
            raise ValueError(f"Unknown step type: {step_type}")
    
    async def _execute_substeps(self, 
                               substeps: List[Dict[str, Any]],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Exécuter des sous-étapes"""
        results = []
        for substep in substeps:
            result = await self._execute_step(substep, context)
            results.append(result)
            context.update(result.get("outputs", {}))
        
        return {"outputs": {"substep_results": results}}
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Évaluer une condition simple"""
        # Implémentation basique - peut être étendue
        try:
            # Remplacer les variables de contexte
            for key, value in context.items():
                condition = condition.replace(f"${{{key}}}", str(value))
            
            # Évaluation sécurisée
            return eval(condition, {"__builtins__": {}}, {})
        except:
            return False


class PluginManager:
    """Gestionnaire de plugins pour extensibilité"""
    
    def __init__(self):
        self.plugins: Dict[str, Any] = {}
        self.plugin_hooks: Dict[str, List[Callable]] = {}
    
    def register_plugin(self, 
                       plugin_name: str,
                       plugin_instance: Any,
                       hooks: Optional[List[str]] = None) -> bool:
        """Enregistrer un plugin"""
        self.plugins[plugin_name] = plugin_instance
        
        # Enregistrer les hooks si spécifiés
        if hooks:
            for hook in hooks:
                if hook not in self.plugin_hooks:
                    self.plugin_hooks[hook] = []
                self.plugin_hooks[hook].append(plugin_instance)
        
        logger.info(f"Registered plugin: {plugin_name}")
        return True
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Désenregistrer un plugin"""
        if plugin_name in self.plugins:
            plugin_instance = self.plugins[plugin_name]
            
            # Supprimer des hooks
            for hook_list in self.plugin_hooks.values():
                if plugin_instance in hook_list:
                    hook_list.remove(plugin_instance)
            
            del self.plugins[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
        return False
    
    async def execute_hook(self, 
                          hook_name: str,
                          *args, **kwargs) -> List[Any]:
        """Exécuter un hook sur tous les plugins enregistrés"""
        results = []
        
        if hook_name in self.plugin_hooks:
            for plugin in self.plugin_hooks[hook_name]:
                try:
                    if hasattr(plugin, hook_name):
                        method = getattr(plugin, hook_name)
                        if asyncio.iscoroutinefunction(method):
                            result = await method(*args, **kwargs)
                        else:
                            result = method(*args, **kwargs)
                        results.append(result)
                except Exception as e:
                    logger.error(f"Plugin hook execution failed: {e}")
                    results.append(None)
        
        return results


class IntegrationConfig(BaseModel):
    """Configuration d'intégration"""
    
    # Configuration générale
    service_name: str = "spotify-ai-integration"
    environment: str = "development"
    debug: bool = True
    
    # Configuration des frameworks
    enabled_frameworks: List[str] = [
        "core", "backend", "microservices", 
        "ml", "monitoring", "security"
    ]
    
    # Configuration du bus d'événements
    event_bus: Dict[str, Any] = {
        "max_history": 1000,
        "enable_persistence": False
    }
    
    # Configuration du registre de services
    service_registry: Dict[str, Any] = {
        "health_check_interval": 30,
        "auto_cleanup": True
    }
    
    # Configuration des workflows
    workflow_engine: Dict[str, Any] = {
        "max_concurrent_workflows": 10,
        "timeout_seconds": 300
    }
    
    # Configuration des plugins
    plugin_manager: Dict[str, Any] = {
        "auto_discover": True,
        "plugin_paths": ["plugins/"]
    }
    
    # Configuration de la base de données
    database: Dict[str, Any] = {
        "url": "postgresql://localhost:5432/spotify_ai",
        "pool_size": 10
    }
    
    # Configuration du cache
    cache: Dict[str, Any] = {
        "redis_url": "redis://localhost:6379/0",
        "default_ttl": 3600
    }
    
    @validator('enabled_frameworks')
    def validate_frameworks(cls, v):
        valid_frameworks = ["core", "backend", "microservices", "ml", "monitoring", "security"]
        for framework in v:
            if framework not in valid_frameworks:
                raise ValueError(f"Invalid framework: {framework}")
        return v


class FrameworkIntegrationManager:
    """Gestionnaire principal d'intégration des frameworks"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.frameworks: Dict[str, FrameworkInfo] = {}
        self.status = IntegrationStatus.INITIALIZING
        
        # Composants d'intégration
        self.event_bus = EventBus()
        self.service_registry = ServiceRegistry()
        self.workflow_engine = WorkflowEngine(self)
        self.plugin_manager = PluginManager()
        
        # Métriques
        self.metrics = {
            "messages_sent": Counter("integration_messages_sent_total", "Total messages sent"),
            "messages_received": Counter("integration_messages_received_total", "Total messages received"),
            "workflow_executions": Counter("workflow_executions_total", "Total workflow executions"),
            "framework_health_checks": Histogram("framework_health_check_duration_seconds", "Framework health check duration")
        }
        
        # Tâches en arrière-plan
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info(f"Initialized FrameworkIntegrationManager with config: {config.service_name}")
    
    async def initialize(self) -> bool:
        """Initialiser le gestionnaire d'intégration"""
        try:
            # Initialiser les frameworks activés
            for framework_name in self.config.enabled_frameworks:
                await self._initialize_framework(framework_name)
            
            # Démarrer les tâches en arrière-plan
            await self._start_background_tasks()
            
            self.status = IntegrationStatus.READY
            logger.info("Framework integration manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize integration manager: {e}")
            self.status = IntegrationStatus.FAILED
            return False
    
    async def start(self) -> bool:
        """Démarrer tous les frameworks intégrés"""
        try:
            # Démarrer les frameworks dans l'ordre des dépendances
            start_order = self._calculate_start_order()
            
            for framework_name in start_order:
                framework_info = self.frameworks[framework_name]
                
                logger.info(f"Starting framework: {framework_name}")
                success = await framework_info.instance.start()
                
                if success:
                    framework_info.status = IntegrationStatus.RUNNING
                    framework_info.startup_time = datetime.utcnow()
                    
                    # Publier événement de démarrage
                    await self._publish_framework_event(
                        framework_name, "started", {"startup_time": framework_info.startup_time}
                    )
                else:
                    framework_info.status = IntegrationStatus.FAILED
                    logger.error(f"Failed to start framework: {framework_name}")
            
            self.status = IntegrationStatus.RUNNING
            logger.info("All frameworks started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start frameworks: {e}")
            self.status = IntegrationStatus.FAILED
            return False
    
    async def stop(self) -> bool:
        """Arrêter tous les frameworks"""
        try:
            # Arrêter les tâches en arrière-plan
            for task in self.background_tasks:
                task.cancel()
            
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            self.background_tasks.clear()
            
            # Arrêter les frameworks dans l'ordre inverse
            stop_order = list(reversed(self._calculate_start_order()))
            
            for framework_name in stop_order:
                framework_info = self.frameworks[framework_name]
                
                logger.info(f"Stopping framework: {framework_name}")
                await framework_info.instance.stop()
                framework_info.status = IntegrationStatus.STOPPED
                
                # Publier événement d'arrêt
                await self._publish_framework_event(
                    framework_name, "stopped", {}
                )
            
            self.status = IntegrationStatus.STOPPED
            logger.info("All frameworks stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop frameworks: {e}")
            return False
    
    async def _initialize_framework(self, framework_name: str) -> bool:
        """Initialiser un framework spécifique"""
        try:
            # Créer l'instance du framework
            framework_instance = await self._create_framework_instance(framework_name)
            
            if not framework_instance:
                logger.error(f"Failed to create framework instance: {framework_name}")
                return False
            
            # Créer les informations du framework
            framework_info = FrameworkInfo(
                name=framework_name,
                type=FrameworkType(framework_name),
                version=getattr(framework_instance, 'version', '1.0.0'),
                instance=framework_instance
            )
            
            # Initialiser le framework
            framework_config = self._get_framework_config(framework_name)
            success = await framework_instance.initialize(framework_config)
            
            if success:
                framework_info.status = IntegrationStatus.READY
                self.frameworks[framework_name] = framework_info
                
                # Enregistrer dans le registre de services
                if hasattr(framework_instance, 'get_endpoints'):
                    endpoints = framework_instance.get_endpoints()
                    for endpoint in endpoints:
                        self.service_registry.register_service(
                            f"{framework_name}_{endpoint['name']}", 
                            endpoint['url'],
                            endpoint.get('metadata', {})
                        )
                
                logger.info(f"Framework initialized successfully: {framework_name}")
                return True
            else:
                logger.error(f"Framework initialization failed: {framework_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing framework {framework_name}: {e}")
            return False
    
    async def _create_framework_instance(self, framework_name: str) -> Optional[Any]:
        """Créer une instance de framework"""
        framework_classes = {
            "core": CoreFramework,
            "backend": HybridBackendFramework,
            "microservices": MicroservicesFramework,
            "ml": MLFrameworksManager,
            "monitoring": MonitoringFramework,
            "security": SecurityFramework
        }
        
        framework_class = framework_classes.get(framework_name)
        if not framework_class:
            logger.error(f"Unknown framework: {framework_name}")
            return None
        
        return framework_class()
    
    def _get_framework_config(self, framework_name: str) -> Dict[str, Any]:
        """Récupérer la configuration d'un framework"""
        # Configuration par défaut
        base_config = {
            "environment": self.config.environment,
            "debug": self.config.debug,
            "service_name": f"{self.config.service_name}-{framework_name}",
            "integration_manager": self
        }
        
        # Configuration spécifique au framework
        framework_configs = {
            "core": {
                "database": self.config.database,
                "cache": self.config.cache
            },
            "backend": {
                "database": self.config.database,
                "cache": self.config.cache
            },
            "microservices": {
                "service_registry": self.service_registry,
                "event_bus": self.event_bus
            },
            "ml": {
                "model_cache": self.config.cache,
                "metrics_enabled": True
            },
            "monitoring": {
                "metrics_port": 8000,
                "health_check_interval": 30
            },
            "security": {
                "jwt_secret": "your-secret-key",
                "token_expiry": 3600
            }
        }
        
        specific_config = framework_configs.get(framework_name, {})
        return {**base_config, **specific_config}
    
    def _calculate_start_order(self) -> List[str]:
        """Calculer l'ordre de démarrage basé sur les dépendances"""
        # Dépendances simples - peut être amélioré avec un algorithme topologique
        dependency_order = [
            "security",     # Base sécurisée
            "core",         # Fonctionnalités de base
            "backend",      # Services backend
            "ml",           # Modèles ML
            "monitoring",   # Surveillance
            "microservices" # Orchestration
        ]
        
        # Filtrer seulement les frameworks activés
        return [fw for fw in dependency_order if fw in self.frameworks]
    
    async def _start_background_tasks(self) -> None:
        """Démarrer les tâches en arrière-plan"""
        # Tâche de vérification de santé
        health_check_task = asyncio.create_task(self._health_check_loop())
        self.background_tasks.append(health_check_task)
        
        # Tâche de nettoyage du registre de services
        cleanup_task = asyncio.create_task(self._service_cleanup_loop())
        self.background_tasks.append(cleanup_task)
        
        # Tâche de métriques
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.append(metrics_task)
        
        logger.info("Background tasks started")
    
    async def _health_check_loop(self) -> None:
        """Boucle de vérification de santé des frameworks"""
        while True:
            try:
                for framework_name, framework_info in self.frameworks.items():
                    if framework_info.status == IntegrationStatus.RUNNING:
                        start_time = datetime.utcnow()
                        
                        try:
                            health_status = await framework_info.instance.health_check()
                            framework_info.last_health_check = datetime.utcnow()
                            
                            # Enregistrer métriques
                            duration = (datetime.utcnow() - start_time).total_seconds()
                            self.metrics["framework_health_checks"].observe(duration)
                            
                            # Mettre à jour le statut dans le registre
                            self.service_registry.update_health_status(
                                framework_name, 
                                health_status.get("healthy", False)
                            )
                            
                        except Exception as e:
                            logger.warning(f"Health check failed for {framework_name}: {e}")
                            framework_info.status = IntegrationStatus.DEGRADED
                            self.service_registry.update_health_status(framework_name, False)
                
                await asyncio.sleep(30)  # Vérification toutes les 30 secondes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)
    
    async def _service_cleanup_loop(self) -> None:
        """Boucle de nettoyage du registre de services"""
        while True:
            try:
                if self.config.service_registry.get("auto_cleanup", True):
                    # Nettoyer les services non sains
                    all_services = self.service_registry.list_services(filter_healthy=False)
                    for service_name in list(all_services.keys()):
                        if service_name not in self.service_registry.health_checks:
                            # Service sans health check récent
                            registered_at = all_services[service_name].get("registered_at")
                            if registered_at and (datetime.utcnow() - registered_at).seconds > 300:
                                self.service_registry.unregister_service(service_name)
                                logger.info(f"Auto-cleaned inactive service: {service_name}")
                
                await asyncio.sleep(60)  # Nettoyage toutes les minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Service cleanup loop error: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_collection_loop(self) -> None:
        """Boucle de collecte de métriques"""
        while True:
            try:
                # Collecter les métriques des frameworks
                for framework_name, framework_info in self.frameworks.items():
                    if framework_info.status == IntegrationStatus.RUNNING:
                        if hasattr(framework_info.instance, 'get_metrics'):
                            framework_metrics = await framework_info.instance.get_metrics()
                            # Traiter les métriques framework-spécifiques
                
                await asyncio.sleep(60)  # Collecte toutes les minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(30)
    
    async def _publish_framework_event(self, 
                                      framework_name: str,
                                      event_type: str,
                                      data: Dict[str, Any]) -> None:
        """Publier un événement de framework"""
        message = IntegrationMessage(
            id=f"fw_{framework_name}_{event_type}_{datetime.utcnow().timestamp()}",
            type=MessageType.EVENT,
            source=framework_name,
            target="*",
            payload={
                "event_type": event_type,
                "data": data
            }
        )
        
        await self.event_bus.publish(message)
    
    # API publique pour l'intégration
    
    def get_framework(self, framework_name: str) -> Optional[FrameworkInfo]:
        """Récupérer les informations d'un framework"""
        return self.frameworks.get(framework_name)
    
    def list_frameworks(self) -> List[FrameworkInfo]:
        """Lister tous les frameworks"""
        return list(self.frameworks.values())
    
    async def send_message(self, message: IntegrationMessage) -> Optional[IntegrationMessage]:
        """Envoyer un message à un framework"""
        target_framework = self.frameworks.get(message.target)
        if not target_framework:
            logger.error(f"Target framework not found: {message.target}")
            return None
        
        try:
            self.metrics["messages_sent"].inc()
            response = await target_framework.instance.handle_message(message)
            
            if response:
                self.metrics["messages_received"].inc()
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to send message to {message.target}: {e}")
            return None
    
    async def broadcast_message(self, message: IntegrationMessage) -> List[Optional[IntegrationMessage]]:
        """Diffuser un message à tous les frameworks"""
        responses = []
        
        for framework_name, framework_info in self.frameworks.items():
            if framework_name != message.source:
                message_copy = IntegrationMessage(
                    id=f"{message.id}_{framework_name}",
                    type=message.type,
                    source=message.source,
                    target=framework_name,
                    payload=message.payload,
                    correlation_id=message.correlation_id
                )
                
                response = await self.send_message(message_copy)
                responses.append(response)
        
        return responses
    
    async def execute_workflow(self, 
                              workflow_id: str,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Exécuter un workflow"""
        self.metrics["workflow_executions"].inc()
        return await self.workflow_engine.execute_workflow(workflow_id, context)
    
    def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Récupérer les informations d'un service"""
        return self.service_registry.discover_service(service_name)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Récupérer le statut global d'intégration"""
        framework_statuses = {
            name: info.status.value 
            for name, info in self.frameworks.items()
        }
        
        return {
            "overall_status": self.status.value,
            "frameworks": framework_statuses,
            "services": len(self.service_registry.list_services()),
            "active_workflows": len(self.workflow_engine.running_workflows),
            "plugins": len(self.plugin_manager.plugins)
        }


# Factory function pour créer le gestionnaire d'intégration
async def create_integration_manager(config: Optional[IntegrationConfig] = None) -> FrameworkIntegrationManager:
    """Créer et initialiser le gestionnaire d'intégration"""
    if config is None:
        config = IntegrationConfig()
    
    manager = FrameworkIntegrationManager(config)
    
    success = await manager.initialize()
    if not success:
        raise RuntimeError("Failed to initialize integration manager")
    
    return manager


# Export des classes principales
__all__ = [
    "FrameworkIntegrationManager",
    "IntegrationConfig", 
    "IntegrationMessage",
    "FrameworkInterface",
    "EventBus",
    "ServiceRegistry",
    "WorkflowEngine",
    "PluginManager",
    "create_integration_manager",
    "IntegrationStatus",
    "FrameworkType",
    "MessageType"
]
