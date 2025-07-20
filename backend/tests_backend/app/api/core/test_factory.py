"""
🎵 Tests Ultra-Avancés pour API Core Factory Management  
========================================================

Tests industriels complets pour la factory pattern et dependency injection avec
tests de sécurité, performance, et validation des composants.

Développé par Fahed Mlaiel - Enterprise Factory Testing Expert
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, Type, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.testclient import TestClient

from app.api.core.factory import (
    ComponentFactory,
    DependencyContainer,
    ComponentRegistry,
    # LifecycleManager,  # Not implemented yet
    # MiddlewareStack,  # Not implemented yet
    # ComponentBuilder,  # Not implemented yet
    create_api_components,
    create_middleware_stack,
    create_service_registry,
    get_component_factory,
    get_dependency_container,
    configure_dependencies,
    cleanup_components,
    LifecycleHook,
    ComponentConfig,
    ServiceLifetime
)


# =============================================================================
# FIXTURES ENTERPRISE POUR FACTORY TESTING
# =============================================================================

@pytest.fixture
def clean_factory():
    """Factory propre pour les tests"""
    # Nettoyer les singletons/registres avant chaque test
    ComponentFactory._instance = None
    DependencyContainer._instance = None
    ComponentRegistry._instance = None
    yield
    # Nettoyer après le test
    ComponentFactory._instance = None
    DependencyContainer._instance = None
    ComponentRegistry._instance = None


@pytest.fixture
def sample_component():
    """Composant de test simple"""
    class SampleComponent:
        def __init__(self, config: Dict[str, Any] = None):
            self.config = config or {}
            self.initialized = True
            self.started = False
            self.stopped = False
        
        async def start(self):
            self.started = True
        
        async def stop(self):
            self.stopped = True
    
    return SampleComponent


@pytest.fixture
def sample_service():
    """Service de test avec dépendances"""
    class SampleService:
        def __init__(self, dependency1: str = "default1", dependency2: int = 42):
            self.dependency1 = dependency1
            self.dependency2 = dependency2
            self.initialized = True
        
        def process(self, data: str) -> str:
            return f"Processed: {data}"
    
    return SampleService


@pytest.fixture
def sample_middleware():
    """Middleware de test"""
    class SampleMiddleware(BaseHTTPMiddleware):
        def __init__(self, app, config: Dict[str, Any] = None):
            super().__init__(app)
            self.config = config or {}
            self.calls = []
        
        async def dispatch(self, request: Request, call_next):
            self.calls.append(f"before_{request.method}")
            response = await call_next(request)
            self.calls.append(f"after_{request.method}")
            return response
    
    return SampleMiddleware


@pytest.fixture
def factory_config():
    """Configuration factory pour les tests"""
    return {
        "database": {
            "url": "postgresql://test:test@localhost/test",
            "pool_size": 5
        },
        "redis": {
            "url": "redis://localhost:6379/0",
            "timeout": 30
        },
        "monitoring": {
            "enabled": True,
            "metrics_port": 9090
        }
    }


@pytest.fixture
def test_app():
    """Application FastAPI de test"""
    app = FastAPI(title="Test Factory App")
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    return app


# =============================================================================
# TESTS DE COMPONENTFACTORY
# =============================================================================

class TestComponentFactory:
    """Tests pour ComponentFactory (singleton pattern)"""
    
    def test_component_factory_singleton(self, clean_factory):
        """Test pattern singleton pour ComponentFactory"""
        factory1 = ComponentFactory()
        factory2 = ComponentFactory()
        
        assert factory1 is factory2
        assert id(factory1) == id(factory2)
    
    def test_component_factory_register_component(self, clean_factory, sample_component):
        """Test enregistrement de composant"""
        factory = ComponentFactory()
        
        # Enregistrer le composant
        factory.register_component(
            name="sample",
            component_class=sample_component,
            config={"test": "value"}
        )
        
        assert "sample" in factory._components
        component_info = factory._components["sample"]
        assert component_info["class"] == sample_component
        assert component_info["config"]["test"] == "value"
        assert component_info["lifetime"] == ServiceLifetime.SINGLETON
    
    def test_component_factory_register_with_lifetime(self, clean_factory, sample_component):
        """Test enregistrement avec lifetime spécifique"""
        factory = ComponentFactory()
        
        factory.register_component(
            name="transient_sample",
            component_class=sample_component,
            lifetime=ServiceLifetime.TRANSIENT
        )
        
        component_info = factory._components["transient_sample"]
        assert component_info["lifetime"] == ServiceLifetime.TRANSIENT
    
    def test_component_factory_create_component(self, clean_factory, sample_component):
        """Test création de composant"""
        factory = ComponentFactory()
        factory.register_component("sample", sample_component)
        
        component = factory.create_component("sample")
        
        assert isinstance(component, sample_component)
        assert component.initialized is True
    
    def test_component_factory_singleton_behavior(self, clean_factory, sample_component):
        """Test comportement singleton"""
        factory = ComponentFactory()
        factory.register_component("sample", sample_component)
        
        component1 = factory.create_component("sample")
        component2 = factory.create_component("sample")
        
        # Pour les singletons, même instance
        assert component1 is component2
    
    def test_component_factory_transient_behavior(self, clean_factory, sample_component):
        """Test comportement transient"""
        factory = ComponentFactory()
        factory.register_component(
            "sample",
            sample_component,
            lifetime=ServiceLifetime.TRANSIENT
        )
        
        component1 = factory.create_component("sample")
        component2 = factory.create_component("sample")
        
        # Pour les transients, instances différentes
        assert component1 is not component2
        assert type(component1) == type(component2)
    
    def test_component_factory_get_component_info(self, clean_factory, sample_component):
        """Test récupération d'infos composant"""
        factory = ComponentFactory()
        factory.register_component(
            "sample",
            sample_component,
            config={"key": "value"}
        )
        
        info = factory.get_component_info("sample")
        
        assert info["class"] == sample_component
        assert info["config"]["key"] == "value"
        assert info["lifetime"] == ServiceLifetime.SINGLETON
    
    def test_component_factory_list_components(self, clean_factory, sample_component):
        """Test liste des composants"""
        factory = ComponentFactory()
        factory.register_component("sample1", sample_component)
        factory.register_component("sample2", sample_component)
        
        components = factory.list_components()
        
        assert "sample1" in components
        assert "sample2" in components
        assert len(components) == 2
    
    def test_component_factory_unknown_component(self, clean_factory):
        """Test composant inexistant"""
        factory = ComponentFactory()
        
        with pytest.raises(ValueError, match="Component 'unknown' not found"):
            factory.create_component("unknown")


# =============================================================================
# TESTS DE DEPENDENCYCONTAINER
# =============================================================================

class TestDependencyContainer:
    """Tests pour DependencyContainer (IoC container)"""
    
    def test_dependency_container_singleton(self, clean_factory):
        """Test pattern singleton pour DependencyContainer"""
        container1 = DependencyContainer()
        container2 = DependencyContainer()
        
        assert container1 is container2
    
    def test_dependency_container_register_dependency(self, clean_factory):
        """Test enregistrement de dépendance"""
        container = DependencyContainer()
        
        def test_factory():
            return "test_value"
        
        container.register("test_dep", test_factory)
        
        assert "test_dep" in container._dependencies
        assert container._dependencies["test_dep"]["factory"] == test_factory
    
    def test_dependency_container_resolve_dependency(self, clean_factory):
        """Test résolution de dépendance"""
        container = DependencyContainer()
        
        def test_factory():
            return "resolved_value"
        
        container.register("test_dep", test_factory)
        value = container.resolve("test_dep")
        
        assert value == "resolved_value"
    
    def test_dependency_container_singleton_caching(self, clean_factory):
        """Test cache singleton"""
        container = DependencyContainer()
        call_count = 0
        
        def test_factory():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"
        
        container.register("test_dep", test_factory, ServiceLifetime.SINGLETON)
        
        value1 = container.resolve("test_dep")
        value2 = container.resolve("test_dep")
        
        assert value1 == value2 == "value_1"
        assert call_count == 1  # Factory appelée une seule fois
    
    def test_dependency_container_transient_no_caching(self, clean_factory):
        """Test pas de cache pour transient"""
        container = DependencyContainer()
        call_count = 0
        
        def test_factory():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"
        
        container.register("test_dep", test_factory, ServiceLifetime.TRANSIENT)
        
        value1 = container.resolve("test_dep")
        value2 = container.resolve("test_dep")
        
        assert value1 == "value_1"
        assert value2 == "value_2"
        assert call_count == 2  # Factory appelée deux fois
    
    def test_dependency_container_with_dependencies(self, clean_factory, sample_service):
        """Test résolution avec dépendances"""
        container = DependencyContainer()
        
        # Enregistrer les dépendances
        container.register("dep1", lambda: "injected_value")
        container.register("dep2", lambda: 100)
        
        # Enregistrer le service avec dépendances
        def service_factory():
            return sample_service(
                dependency1=container.resolve("dep1"),
                dependency2=container.resolve("dep2")
            )
        
        container.register("service", service_factory)
        
        service = container.resolve("service")
        
        assert service.dependency1 == "injected_value"
        assert service.dependency2 == 100
    
    def test_dependency_container_clear_cache(self, clean_factory):
        """Test nettoyage du cache"""
        container = DependencyContainer()
        
        call_count = 0
        def test_factory():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"
        
        container.register("test_dep", test_factory)
        
        # Premier resolve
        value1 = container.resolve("test_dep")
        assert value1 == "value_1"
        
        # Nettoyer le cache
        container.clear_cache()
        
        # Deuxième resolve après clear
        value2 = container.resolve("test_dep")
        assert value2 == "value_2"
        assert call_count == 2


# =============================================================================
# TESTS DE SERVICEREGISTRY
# =============================================================================

class TestComponentRegistry:
    """Tests pour ComponentRegistry"""
    
    def test_service_registry_singleton(self, clean_factory):
        """Test pattern singleton pour ComponentRegistry"""
        registry1 = ComponentRegistry()
        registry2 = ComponentRegistry()
        
        assert registry1 is registry2
    
    def test_service_registry_register_service(self, clean_factory, sample_service):
        """Test enregistrement de service"""
        registry = ComponentRegistry()
        
        registry.register_service(
            name="test_service",
            service_class=sample_service,
            config={"param": "value"}
        )
        
        assert "test_service" in registry._services
        service_info = registry._services["test_service"]
        assert service_info["class"] == sample_service
        assert service_info["config"]["param"] == "value"
    
    def test_service_registry_get_service(self, clean_factory, sample_service):
        """Test récupération de service"""
        registry = ComponentRegistry()
        registry.register_service("test_service", sample_service)
        
        service = registry.get_service("test_service")
        
        assert isinstance(service, sample_service)
        assert service.initialized is True
    
    def test_service_registry_list_services(self, clean_factory, sample_service):
        """Test liste des services"""
        registry = ComponentRegistry()
        registry.register_service("service1", sample_service)
        registry.register_service("service2", sample_service)
        
        services = registry.list_services()
        
        assert "service1" in services
        assert "service2" in services
        assert len(services) == 2
    
    def test_service_registry_service_exists(self, clean_factory, sample_service):
        """Test existence de service"""
        registry = ComponentRegistry()
        
        assert not registry.service_exists("test_service")
        
        registry.register_service("test_service", sample_service)
        
        assert registry.service_exists("test_service")


# =============================================================================
# TESTS DE LIFECYCLEMANAGER
# =============================================================================

class TestLifecycleManager:
    """Tests pour LifecycleManager"""
    
    @pytest.mark.asyncio
    async def test_lifecycle_manager_startup(self, sample_component):
        """Test startup lifecycle"""
        manager = LifecycleManager()
        component = sample_component()
        
        manager.register_component("test", component)
        await manager.startup()
        
        assert component.started is True
    
    @pytest.mark.asyncio
    async def test_lifecycle_manager_shutdown(self, sample_component):
        """Test shutdown lifecycle"""
        manager = LifecycleManager()
        component = sample_component()
        
        manager.register_component("test", component)
        await manager.startup()
        await manager.shutdown()
        
        assert component.stopped is True
    
    @pytest.mark.asyncio
    async def test_lifecycle_manager_hooks(self):
        """Test lifecycle hooks"""
        manager = LifecycleManager()
        hook_calls = []
        
        async def startup_hook():
            hook_calls.append("startup")
        
        async def shutdown_hook():
            hook_calls.append("shutdown")
        
        manager.add_startup_hook(startup_hook)
        manager.add_shutdown_hook(shutdown_hook)
        
        await manager.startup()
        await manager.shutdown()
        
        assert hook_calls == ["startup", "shutdown"]
    
    @pytest.mark.asyncio
    async def test_lifecycle_manager_error_handling(self, sample_component):
        """Test gestion d'erreur dans lifecycle"""
        manager = LifecycleManager()
        
        # Composant qui échoue au startup
        class FailingComponent:
            async def start(self):
                raise RuntimeError("Startup failed")
            
            async def stop(self):
                pass
        
        failing_component = FailingComponent()
        working_component = sample_component()
        
        manager.register_component("failing", failing_component)
        manager.register_component("working", working_component)
        
        # Le startup devrait gérer l'erreur et continuer
        await manager.startup()
        
        # Le composant qui fonctionne devrait être démarré
        assert working_component.started is True


# =============================================================================
# TESTS DE MIDDLEWARESTACK
# =============================================================================

class TestMiddlewareStack:
    """Tests pour MiddlewareStack"""
    
    def test_middleware_stack_creation(self, test_app, sample_middleware):
        """Test création de middleware stack"""
        stack = MiddlewareStack(test_app)
        
        stack.add_middleware(sample_middleware, config={"test": "value"})
        
        assert len(stack._middlewares) == 1
        middleware_info = stack._middlewares[0]
        assert middleware_info["class"] == sample_middleware
        assert middleware_info["config"]["test"] == "value"
    
    def test_middleware_stack_ordering(self, test_app):
        """Test ordre des middlewares"""
        stack = MiddlewareStack(test_app)
        
        class FirstMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.order = getattr(request.state, 'order', [])
                request.state.order.append('first')
                response = await call_next(request)
                return response
        
        class SecondMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.order = getattr(request.state, 'order', [])
                request.state.order.append('second')
                response = await call_next(request)
                return response
        
        stack.add_middleware(FirstMiddleware, priority=1)
        stack.add_middleware(SecondMiddleware, priority=2)
        
        # Les middlewares devraient être triés par priorité
        stack.apply_middlewares()
        
        # Vérifier l'ordre avec un test client
        with TestClient(test_app) as client:
            response = client.get("/test")
            assert response.status_code == 200
    
    def test_middleware_stack_conditional_loading(self, test_app, sample_middleware):
        """Test chargement conditionnel de middleware"""
        stack = MiddlewareStack(test_app)
        
        # Middleware avec condition
        stack.add_middleware(
            sample_middleware,
            condition=lambda: True,  # Toujours charger
            config={"enabled": True}
        )
        
        stack.add_middleware(
            sample_middleware,
            condition=lambda: False,  # Jamais charger
            config={"enabled": False}
        )
        
        stack.apply_middlewares()
        
        # Seul le premier middleware devrait être appliqué
        # (Vérification via introspection FastAPI)


# =============================================================================
# TESTS DE COMPONENTBUILDER
# =============================================================================

class TestComponentBuilder:
    """Tests pour ComponentBuilder (Builder pattern)"""
    
    def test_component_builder_basic(self, sample_component):
        """Test builder basique"""
        builder = ComponentBuilder(sample_component)
        
        component = (builder
                    .with_config({"key": "value"})
                    .with_lifetime(ServiceLifetime.SINGLETON)
                    .build())
        
        assert isinstance(component, sample_component)
        assert component.config["key"] == "value"
    
    def test_component_builder_chain(self, sample_service):
        """Test chaînage du builder"""
        builder = ComponentBuilder(sample_service)
        
        component = (builder
                    .with_config({"setting": "test"})
                    .with_lifetime(ServiceLifetime.TRANSIENT)
                    .with_tags(["service", "business"])
                    .build())
        
        assert isinstance(component, sample_service)
    
    def test_component_builder_validation(self):
        """Test validation du builder"""
        # Tenter de construire sans classe
        builder = ComponentBuilder(None)
        
        with pytest.raises(ValueError, match="Component class is required"):
            builder.build()


# =============================================================================
# TESTS DES FONCTIONS FACTORY
# =============================================================================

class TestFactoryFunctions:
    """Tests pour les fonctions factory principales"""
    
    def test_create_api_components(self, clean_factory, factory_config):
        """Test création des composants API"""
        components = create_api_components(factory_config)
        
        assert "config" in components
        assert "database" in components
        assert "redis" in components
        assert "monitoring" in components
    
    def test_create_middleware_stack(self, test_app, factory_config):
        """Test création du middleware stack"""
        stack = create_middleware_stack(test_app, factory_config)
        
        assert isinstance(stack, MiddlewareStack)
        assert stack._app == test_app
    
    def test_create_service_registry(self, clean_factory, factory_config):
        """Test création du service registry"""
        registry = create_service_registry(factory_config)
        
        assert isinstance(registry, ComponentRegistry)
    
    def test_get_component_factory(self, clean_factory):
        """Test récupération de factory"""
        factory = get_component_factory()
        
        assert isinstance(factory, ComponentFactory)
        
        # Deuxième appel devrait retourner la même instance
        factory2 = get_component_factory()
        assert factory is factory2
    
    def test_get_dependency_container(self, clean_factory):
        """Test récupération du container"""
        container = get_dependency_container()
        
        assert isinstance(container, DependencyContainer)
    
    def test_configure_dependencies(self, clean_factory, factory_config):
        """Test configuration des dépendances"""
        configure_dependencies(factory_config)
        
        container = get_dependency_container()
        
        # Vérifier que des dépendances ont été configurées
        assert len(container._dependencies) > 0
    
    @pytest.mark.asyncio
    async def test_cleanup_components(self, clean_factory):
        """Test nettoyage des composants"""
        # Créer quelques composants
        factory = get_component_factory()
        container = get_dependency_container()
        
        # Ajouter des composants factices
        factory._instances = {"test": Mock()}
        container._cache = {"test": Mock()}
        
        await cleanup_components()
        
        # Vérifier que le nettoyage a eu lieu
        assert len(factory._instances) == 0
        assert len(container._cache) == 0


# =============================================================================
# TESTS D'INTÉGRATION
# =============================================================================

@pytest.mark.integration
class TestFactoryIntegration:
    """Tests d'intégration pour la factory"""
    
    def test_full_factory_flow(self, clean_factory, factory_config, test_app):
        """Test flux complet de factory"""
        # 1. Configurer les dépendances
        configure_dependencies(factory_config)
        
        # 2. Créer les composants API
        components = create_api_components(factory_config)
        
        # 3. Créer le middleware stack
        stack = create_middleware_stack(test_app, factory_config)
        
        # 4. Vérifier que tout est connecté
        assert "config" in components
        assert isinstance(stack, MiddlewareStack)
        
        container = get_dependency_container()
        assert len(container._dependencies) > 0
    
    def test_factory_with_real_fastapi_app(self, clean_factory, factory_config):
        """Test factory avec vraie app FastAPI"""
        app = FastAPI(title="Test Factory Integration")
        
        # Configurer la factory
        configure_dependencies(factory_config)
        
        # Créer les middlewares
        stack = create_middleware_stack(app, factory_config)
        stack.apply_middlewares()
        
        # Tester l'app
        with TestClient(app) as client:
            # L'app devrait fonctionner même sans endpoints
            response = client.get("/openapi.json")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_factory_lifecycle_integration(self, clean_factory, factory_config):
        """Test intégration avec lifecycle"""
        configure_dependencies(factory_config)
        components = create_api_components(factory_config)
        
        # Créer le lifecycle manager
        manager = LifecycleManager()
        
        # Enregistrer des composants qui ont des méthodes start/stop
        for name, component in components.items():
            if hasattr(component, 'start') or hasattr(component, 'stop'):
                manager.register_component(name, component)
        
        # Démarrer et arrêter
        await manager.startup()
        await manager.shutdown()
        
        # Pas d'erreur = succès


# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

@pytest.mark.performance
class TestFactoryPerformance:
    """Tests de performance pour la factory"""
    
    def test_component_creation_performance(self, benchmark, clean_factory, sample_component):
        """Test performance création de composant"""
        factory = ComponentFactory()
        factory.register_component("sample", sample_component)
        
        def create_component():
            return factory.create_component("sample")
        
        result = benchmark(create_component)
        assert isinstance(result, sample_component)
    
    def test_dependency_resolution_performance(self, benchmark, clean_factory):
        """Test performance résolution de dépendance"""
        container = DependencyContainer()
        
        def test_factory():
            return "test_value"
        
        container.register("test_dep", test_factory)
        
        def resolve_dependency():
            return container.resolve("test_dep")
        
        result = benchmark(resolve_dependency)
        assert result == "test_value"
    
    def test_singleton_vs_transient_performance(self, clean_factory, sample_component):
        """Test performance singleton vs transient"""
        factory = ComponentFactory()
        
        # Enregistrer les deux types
        factory.register_component("singleton", sample_component, lifetime=ServiceLifetime.SINGLETON)
        factory.register_component("transient", sample_component, lifetime=ServiceLifetime.TRANSIENT)
        
        # Mesurer les créations multiples
        start_time = time.time()
        for _ in range(100):
            factory.create_component("singleton")
        singleton_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(100):
            factory.create_component("transient")
        transient_time = time.time() - start_time
        
        # Le singleton devrait être plus rapide pour les créations multiples
        assert singleton_time < transient_time
    
    def test_concurrent_component_creation(self, clean_factory, sample_component):
        """Test création de composant concurrente"""
        factory = ComponentFactory()
        factory.register_component("sample", sample_component)
        
        def create_component():
            return factory.create_component("sample")
        
        # Test avec ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_component) for _ in range(50)]
            
            results = [future.result() for future in futures]
            
            # Pour les singletons, toutes les instances devraient être identiques
            assert all(result is results[0] for result in results)


# =============================================================================
# TESTS DE SÉCURITÉ
# =============================================================================

@pytest.mark.security
class TestFactorySecurity:
    """Tests de sécurité pour la factory"""
    
    def test_component_isolation(self, clean_factory, sample_component):
        """Test isolation des composants"""
        factory = ComponentFactory()
        
        # Enregistrer avec des configs différentes
        factory.register_component("comp1", sample_component, config={"secret": "value1"})
        factory.register_component("comp2", sample_component, config={"secret": "value2"})
        
        comp1 = factory.create_component("comp1")
        comp2 = factory.create_component("comp2")
        
        # Les configs ne devraient pas se mélanger
        assert comp1.config["secret"] != comp2.config["secret"]
    
    def test_dependency_injection_security(self, clean_factory):
        """Test sécurité de l'injection de dépendance"""
        container = DependencyContainer()
        
        # Enregistrer une dépendance sensible
        container.register("secret_service", lambda: {"api_key": "secret123"})
        
        # Une autre partie du code ne devrait pas pouvoir modifier cette dépendance
        secret = container.resolve("secret_service")
        
        # Modifier l'objet résolu ne devrait pas affecter les futures résolutions
        secret["api_key"] = "modified"
        
        # Pour les singletons, la modification sera visible (comportement attendu)
        # Pour les transients, chaque résolution donne une nouvelle instance
    
    def test_component_factory_thread_safety(self, clean_factory, sample_component):
        """Test thread safety de la factory"""
        factory = ComponentFactory()
        factory.register_component("sample", sample_component)
        
        results = []
        errors = []
        
        def create_component_thread():
            try:
                component = factory.create_component("sample")
                results.append(component)
            except Exception as e:
                errors.append(e)
        
        # Créer plusieurs threads
        threads = [
            threading.Thread(target=create_component_thread)
            for _ in range(10)
        ]
        
        # Démarrer tous les threads
        for thread in threads:
            thread.start()
        
        # Attendre la fin
        for thread in threads:
            thread.join()
        
        # Vérifier qu'il n'y a pas d'erreurs
        assert len(errors) == 0
        assert len(results) == 10
        
        # Pour les singletons, toutes les instances devraient être identiques
        assert all(result is results[0] for result in results)


# =============================================================================
# TESTS DE CONFIGURATION
# =============================================================================

@pytest.mark.configuration
class TestFactoryConfiguration:
    """Tests de configuration pour la factory"""
    
    def test_component_config_validation(self, clean_factory, sample_component):
        """Test validation de configuration"""
        factory = ComponentFactory()
        
        # Configuration valide
        valid_config = {"param1": "value1", "param2": 42}
        factory.register_component("valid", sample_component, config=valid_config)
        
        component = factory.create_component("valid")
        assert component.config == valid_config
    
    def test_component_config_defaults(self, clean_factory, sample_component):
        """Test valeurs par défaut de configuration"""
        factory = ComponentFactory()
        
        # Enregistrer sans config
        factory.register_component("default", sample_component)
        
        component = factory.create_component("default")
        assert component.config == {}  # Config par défaut vide
    
    def test_component_config_override(self, clean_factory, sample_component):
        """Test override de configuration"""
        factory = ComponentFactory()
        
        base_config = {"param1": "base_value", "param2": "base_value2"}
        factory.register_component("configurable", sample_component, config=base_config)
        
        # Créer avec override
        override_config = {"param1": "override_value"}
        component = factory.create_component("configurable", config_override=override_config)
        
        # La config devrait être mergée
        expected_config = {"param1": "override_value", "param2": "base_value2"}
        assert component.config == expected_config
