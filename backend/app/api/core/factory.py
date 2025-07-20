"""
🎵 Spotify AI Agent - API Factory Patterns
==========================================

Factory patterns enterprise pour la création de composants API,
middlewares, et services avec injection de dépendances.

Architecture:
- Abstract Factory pour composants API
- Builder pattern pour configuration complexe
- Dependency Injection container
- Service locator pattern
- Factory method pattern
- Singleton management

Développé par Fahed Mlaiel - Enterprise Factory Pattern Expert
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar, Callable, Optional, List
from functools import lru_cache
from enum import Enum
from dataclasses import dataclass, field
from dataclasses import dataclass, field

from fastapi import FastAPI, Depends
from starlette.middleware.base import BaseHTTPMiddleware

from .config import APISettings, get_settings
from .context import RequestContextMiddleware, APIContext


T = TypeVar('T')
ServiceType = TypeVar('ServiceType')


class ComponentType(str, Enum):
    """Types de composants disponibles"""
    MIDDLEWARE = "middleware"
    SERVICE = "service"
    REPOSITORY = "repository"
    CONTROLLER = "controller"
    VALIDATOR = "validator"
    SERIALIZER = "serializer"
    CACHE = "cache"
    DATABASE = "database"


class LifecycleType(str, Enum):
    """Types de cycle de vie des composants"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"  # Per request
    PROTOTYPE = "prototype"  # New instance each time


class ComponentRegistry:
    """Registre des composants avec gestion du cycle de vie"""
    
    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._instances: Dict[str, Any] = {}
        self._lifecycles: Dict[str, LifecycleType] = {}
        self._dependencies: Dict[str, List[str]] = {}
    
    def register(
        self,
        name: str,
        factory: Callable,
        lifecycle: LifecycleType = LifecycleType.SINGLETON,
        dependencies: List[str] = None
    ):
        """Enregistre un composant"""
        self._factories[name] = factory
        self._lifecycles[name] = lifecycle
        self._dependencies[name] = dependencies or []
    
    def get(self, name: str, **kwargs) -> Any:
        """Récupère une instance de composant"""
        if name not in self._factories:
            raise ValueError(f"Component '{name}' not registered")
        
        lifecycle = self._lifecycles[name]
        
        # Singleton: une seule instance
        if lifecycle == LifecycleType.SINGLETON:
            if name not in self._instances:
                self._instances[name] = self._create_instance(name, **kwargs)
            return self._instances[name]
        
        # Transient: nouvelle instance à chaque fois
        elif lifecycle == LifecycleType.TRANSIENT:
            return self._create_instance(name, **kwargs)
        
        # Scoped: instance par requête (TODO: implémenter avec context)
        elif lifecycle == LifecycleType.SCOPED:
            # Pour l'instant, comportement transient
            return self._create_instance(name, **kwargs)
        
        # Prototype: nouvelle instance configurée
        else:
            return self._create_instance(name, **kwargs)
    
    def _create_instance(self, name: str, **kwargs) -> Any:
        """Crée une instance avec injection de dépendances"""
        factory = self._factories[name]
        dependencies = self._dependencies[name]
        
        # Résoudre les dépendances
        dep_instances = {}
        for dep_name in dependencies:
            dep_instances[dep_name] = self.get(dep_name)
        
        # Merger avec les kwargs fournis
        all_kwargs = {**dep_instances, **kwargs}
        
        # Inspecter la signature pour ne passer que les params attendus
        sig = inspect.signature(factory)
        filtered_kwargs = {
            k: v for k, v in all_kwargs.items() 
            if k in sig.parameters
        }
        
        return factory(**filtered_kwargs)
    
    def is_registered(self, name: str) -> bool:
        """Vérifie si un composant est enregistré"""
        return name in self._factories
    
    def clear(self):
        """Vide le registre"""
        self._factories.clear()
        self._instances.clear()
        self._lifecycles.clear()
        self._dependencies.clear()


class ComponentFactory(ABC):
    """Factory abstrait pour les composants"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
    
    @abstractmethod
    def create(self, name: str, **kwargs) -> Any:
        """Crée un composant"""
        pass
    
    @abstractmethod
    def register_defaults(self):
        """Enregistre les composants par défaut"""
        pass


class MiddlewareFactory(ComponentFactory):
    """Factory pour les middlewares"""
    
    def create(self, name: str, **kwargs) -> BaseHTTPMiddleware:
        """Crée un middleware"""
        return self.registry.get(name, **kwargs)
    
    def register_defaults(self):
        """Enregistre les middlewares par défaut"""
        from app.api.middleware.cache_middleware import AdvancedCacheMiddleware
        from app.api.middleware.auth_middleware import AuthenticationMiddleware
        from app.api.middleware.cors_middleware import CORSMiddleware
        
        # Cache Middleware
        self.registry.register(
            "cache_middleware",
            lambda config=None: AdvancedCacheMiddleware(config),
            LifecycleType.SINGLETON
        )
        
        # Auth Middleware  
        self.registry.register(
            "auth_middleware",
            lambda config=None: AuthenticationMiddleware(config),
            LifecycleType.SINGLETON
        )
        
        # Context Middleware
        self.registry.register(
            "context_middleware",
            lambda api_context=None: RequestContextMiddleware(api_context),
            LifecycleType.SINGLETON
        )


class ServiceFactory(ComponentFactory):
    """Factory pour les services métier"""
    
    def create(self, name: str, **kwargs) -> Any:
        """Crée un service"""
        return self.registry.get(name, **kwargs)
    
    def register_defaults(self):
        """Enregistre les services par défaut"""
        # TODO: Ajouter les services quand ils seront disponibles
        pass


class DatabaseFactory(ComponentFactory):
    """Factory pour les composants de base de données"""
    
    def create(self, name: str, **kwargs) -> Any:
        """Crée un composant database"""
        return self.registry.get(name, **kwargs)
    
    def register_defaults(self):
        """Enregistre les composants database par défaut"""
        from app.core.database import get_database_pool
        
        self.registry.register(
            "database_pool",
            get_database_pool,
            LifecycleType.SINGLETON
        )


class CacheFactory(ComponentFactory):
    """Factory pour les composants de cache"""
    
    def create(self, name: str, **kwargs) -> Any:
        """Crée un composant cache"""
        return self.registry.get(name, **kwargs)
    
    def register_defaults(self):
        """Enregistre les composants cache par défaut"""
        from app.utils.cache.manager import AdvancedCacheManager
        
        self.registry.register(
            "cache_manager", 
            lambda config=None: AdvancedCacheManager(config),
            LifecycleType.SINGLETON
        )


# =============================================================================
# CONTAINER D'INJECTION DE DÉPENDANCES
# =============================================================================

class DependencyContainer:
    """Container pour l'injection de dépendances"""
    
    def __init__(self):
        self.registry = ComponentRegistry()
        self.factories = {
            ComponentType.MIDDLEWARE: MiddlewareFactory(self.registry),
            ComponentType.SERVICE: ServiceFactory(self.registry),
            ComponentType.DATABASE: DatabaseFactory(self.registry),
            ComponentType.CACHE: CacheFactory(self.registry)
        }
        
        # Enregistrer les composants par défaut
        self._register_defaults()
    
    def _register_defaults(self):
        """Enregistre tous les composants par défaut"""
        for factory in self.factories.values():
            factory.register_defaults()
    
    def get(self, name: str, component_type: ComponentType = None) -> Any:
        """Récupère un composant"""
        if component_type and component_type in self.factories:
            return self.factories[component_type].create(name)
        else:
            return self.registry.get(name)
    
    def register(
        self,
        name: str,
        factory: Callable,
        component_type: ComponentType = None,
        lifecycle: LifecycleType = LifecycleType.SINGLETON,
        dependencies: List[str] = None
    ):
        """Enregistre un nouveau composant"""
        self.registry.register(name, factory, lifecycle, dependencies)
    
    def create_middleware_stack(self, app: FastAPI, middleware_names: List[str]) -> FastAPI:
        """Crée une pile de middlewares"""
        for name in reversed(middleware_names):  # Ordre inverse pour FastAPI
            middleware = self.get(name, ComponentType.MIDDLEWARE)
            app.add_middleware(type(middleware))
        return app


# =============================================================================
# INSTANCES GLOBALES
# =============================================================================

_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Retourne le container global (Singleton)"""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container


def get_component(name: str, component_type: ComponentType = None) -> Any:
    """Récupère un composant depuis le container global"""
    return get_container().get(name, component_type)


def register_component(
    name: str,
    factory: Callable,
    component_type: ComponentType = None,
    lifecycle: LifecycleType = LifecycleType.SINGLETON,
    dependencies: List[str] = None
):
    """Enregistre un composant dans le container global"""
    get_container().register(name, factory, component_type, lifecycle, dependencies)


# =============================================================================
# FONCTIONS UTILITAIRES DE CRÉATION
# =============================================================================

def create_api_component(
    component_type: ComponentType,
    name: str,
    **kwargs
) -> Any:
    """Crée un composant API"""
    container = get_container()
    return container.get(name, component_type)


def create_middleware_stack(
    app: FastAPI,
    settings: APISettings = None
) -> FastAPI:
    """Crée la pile complète de middlewares"""
    if settings is None:
        settings = get_settings()
    
    container = get_container()
    enabled_middleware = [
        name for name, enabled in settings.api.middleware_enabled.items()
        if enabled
    ]
    
    # Ajouter les middlewares dans l'ordre approprié
    middleware_order = [
        "context_middleware",
        "auth_middleware", 
        "cache_middleware"
    ]
    
    active_middleware = [
        name for name in middleware_order
        if any(enabled_name in name for enabled_name in enabled_middleware)
    ]
    
    return container.create_middleware_stack(app, active_middleware)


def create_fastapi_app(settings: APISettings = None) -> FastAPI:
    """Crée une application FastAPI complète avec tous les composants"""
    if settings is None:
        settings = get_settings()
    
    # Créer l'app FastAPI
    app = FastAPI(
        title=settings.api.app_name,
        version=settings.api.app_version,
        description=settings.api.app_description,
        docs_url=settings.api.docs_url,
        redoc_url=settings.api.redoc_url,
        openapi_url=settings.api.openapi_url
    )
    
    # Ajouter les middlewares
    app = create_middleware_stack(app, settings)
    
    return app


@lru_cache()
def get_cached_component(name: str, component_type: str = None) -> Any:
    """Version cachée de get_component pour les dépendances FastAPI"""
    return get_component(name, ComponentType(component_type) if component_type else None)


# =============================================================================
# DÉCORATEURS POUR INJECTION DE DÉPENDANCES
# =============================================================================

def inject(component_name: str, component_type: ComponentType = None):
    """Décorateur pour injecter des dépendances"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            component = get_component(component_name, component_type)
            return func(component, *args, **kwargs)
        return wrapper
    return decorator


def injectable(
    name: str,
    component_type: ComponentType = None,
    lifecycle: LifecycleType = LifecycleType.SINGLETON
):
    """Décorateur pour marquer une classe comme injectable"""
    def decorator(cls):
        register_component(name, cls, component_type, lifecycle)
        return cls
    return decorator


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ComponentType",
    "LifecycleType", 
    "ComponentRegistry",
    "ComponentFactory",
    "MiddlewareFactory",
    "ServiceFactory",
    "DatabaseFactory",
    "CacheFactory",
    "DependencyContainer",
    "get_container",
    "get_component",
    "register_component",
    "create_api_component",
    "create_api_components",
    "create_middleware_stack",
    "create_fastapi_app",
    "get_cached_component",
    "inject",
    "injectable",
    "create_service_registry",
    "get_component_factory",
    "get_dependency_container",
    "configure_dependencies",
    "cleanup_components",
    "LifecycleHook",
    "ComponentConfig",
    "ServiceLifetime"
]


# =============================================================================
# FONCTIONS UTILITAIRES COMPLÉMENTAIRES
# =============================================================================

def create_api_components(app: FastAPI, settings: APISettings = None) -> Dict[str, Any]:
    """Créer tous les composants API nécessaires"""
    if settings is None:
        settings = get_settings()
    
    container = get_container()
    
    # Créer les composants principaux
    components = {
        "context": container.get_component("api_context"),
        "middleware": create_middleware_stack(app, settings),
        "cache": container.get_component("cache_manager"),
        "database": container.get_component("database_manager"),
    }
    
    return components


def create_service_registry():
    """Créer un registre de services"""
    return ComponentRegistry()


def get_component_factory():
    """Obtenir la factory de composants"""
    return ComponentFactory()


def get_dependency_container():
    """Obtenir le container de dépendances"""
    return get_container()


def configure_dependencies(container, config=None):
    """Configurer les dépendances dans le container"""
    if config is None:
        config = {}
    
    # Configuration par défaut
    container.register("api_context", APIContext, LifecycleType.SINGLETON)
    container.register("cache_manager", dict, LifecycleType.SINGLETON)  # Mock
    container.register("database_manager", dict, LifecycleType.SINGLETON)  # Mock
    
    return container


def cleanup_components():
    """Nettoyer les composants"""
    container = get_container()
    container._instances.clear()
    container._factories.clear()


class LifecycleHook:
    """Hook de cycle de vie pour les composants"""
    
    def __init__(self, name: str, callback: Callable = None):
        self.name = name
        self.callback = callback or (lambda: None)
    
    def execute(self):
        return self.callback()


@dataclass
class ComponentConfig:
    """Configuration d'un composant"""
    name: str
    component_type: ComponentType = ComponentType.SERVICE
    lifecycle: LifecycleType = LifecycleType.SINGLETON
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class ServiceLifetime:
    """Gestion de la durée de vie des services"""
    SINGLETON = LifecycleType.SINGLETON
    TRANSIENT = LifecycleType.TRANSIENT
    SCOPED = LifecycleType.SCOPED
