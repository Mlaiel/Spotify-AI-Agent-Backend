"""
🧪 Tests Framework Orchestrator - Core Engine
============================================

Tests complets du framework orchestrator central avec:
- Gestion du cycle de vie
- Health monitoring
- Circuit breaker patterns
- Dependency management
- Graceful shutdown

Développé par: Lead Developer + AI Architect
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
import time

from backend.app.frameworks.core import (
    BaseFramework,
    FrameworkOrchestrator, 
    FrameworkStatus,
    HealthStatus,
    CircuitBreaker,
    CircuitBreakerState
)
from backend.app.frameworks import TEST_CONFIG, clean_frameworks, logger


class MockFramework(BaseFramework):
    """Framework mock pour les tests."""
    
    def __init__(self, name: str, fail_init: bool = False, fail_health: bool = False):
        super().__init__()
        self.name = name
        self.fail_init = fail_init
        self.fail_health = fail_health
        self.init_called = False
        self.shutdown_called = False
        
    async def initialize(self) -> bool:
        """Initialisation mockée."""
        self.init_called = True
        if self.fail_init:
            raise Exception(f"Init failed for {self.name}")
        self.status = FrameworkStatus.RUNNING
        return True
        
    async def shutdown(self) -> bool:
        """Arrêt mocké."""
        self.shutdown_called = True
        self.status = FrameworkStatus.STOPPED
        return True
        
    async def health_check(self) -> HealthStatus:
        """Health check mocké."""
        if self.fail_health:
            return HealthStatus(
                status=FrameworkStatus.ERROR,
                message=f"Health check failed for {self.name}",
                details={"error": "mock_error"}
            )
        return HealthStatus(
            status=FrameworkStatus.RUNNING,
            message=f"{self.name} is healthy",
            details={"uptime": time.time()}
        )


@pytest.mark.core
class TestBaseFramework:
    """Tests de la classe BaseFramework."""
    
    def test_base_framework_creation(self):
        """Test création framework de base."""
        framework = BaseFramework()
        assert framework.status == FrameworkStatus.STOPPED
        assert framework.metrics == {}
        assert framework.dependencies == []
        
    @pytest.mark.asyncio
    async def test_base_framework_not_implemented(self):
        """Test méthodes non implémentées."""
        framework = BaseFramework()
        
        with pytest.raises(NotImplementedError):
            await framework.initialize()
            
        with pytest.raises(NotImplementedError):
            await framework.shutdown()
            
        with pytest.raises(NotImplementedError):
            await framework.health_check()


@pytest.mark.core
class TestCircuitBreaker:
    """Tests du Circuit Breaker."""
    
    def test_circuit_breaker_creation(self):
        """Test création circuit breaker."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=Exception
        )
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker avec succès."""
        cb = CircuitBreaker(failure_threshold=3)
        
        async def successful_func():
            return "success"
            
        result = await cb.call(successful_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self):
        """Test seuil d'échec du circuit breaker."""
        cb = CircuitBreaker(failure_threshold=2)
        
        async def failing_func():
            raise Exception("Test failure")
            
        # Premier échec
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 1
        
        # Deuxième échec - circuit breaker s'ouvre
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 2
        
        # Tentative suivante - circuit ouvert
        with pytest.raises(Exception, match="Circuit breaker is open"):
            await cb.call(failing_func)
            
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test récupération en état half-open."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        async def failing_func():
            raise Exception("Test failure")
            
        async def successful_func():
            return "recovered"
            
        # Échec - circuit s'ouvre
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.state == CircuitBreakerState.OPEN
        
        # Attendre timeout
        await asyncio.sleep(0.2)
        
        # Succès en half-open - circuit se ferme
        result = await cb.call(successful_func)
        assert result == "recovered"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0


@pytest.mark.core
class TestFrameworkOrchestrator:
    """Tests du Framework Orchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_singleton(self, clean_frameworks):
        """Test pattern singleton de l'orchestrator."""
        orchestrator1 = FrameworkOrchestrator()
        orchestrator2 = FrameworkOrchestrator()
        assert orchestrator1 is orchestrator2
        
    @pytest.mark.asyncio
    async def test_register_framework(self, clean_frameworks):
        """Test enregistrement d'un framework."""
        orchestrator = FrameworkOrchestrator()
        framework = MockFramework("test_framework")
        
        orchestrator.register_framework("test", framework)
        assert "test" in orchestrator.frameworks
        assert orchestrator.frameworks["test"] is framework
        
    @pytest.mark.asyncio
    async def test_initialize_single_framework(self, clean_frameworks):
        """Test initialisation d'un framework unique."""
        orchestrator = FrameworkOrchestrator()
        framework = MockFramework("test_framework")
        orchestrator.register_framework("test", framework)
        
        result = await orchestrator.initialize_framework("test")
        assert result is True
        assert framework.init_called is True
        assert framework.status == FrameworkStatus.RUNNING
        
    @pytest.mark.asyncio
    async def test_initialize_framework_failure(self, clean_frameworks):
        """Test échec d'initialisation d'un framework."""
        orchestrator = FrameworkOrchestrator()
        framework = MockFramework("test_framework", fail_init=True)
        orchestrator.register_framework("test", framework)
        
        result = await orchestrator.initialize_framework("test")
        assert result is False
        assert framework.init_called is True
        assert framework.status == FrameworkStatus.ERROR
        
    @pytest.mark.asyncio
    async def test_initialize_all_frameworks(self, clean_frameworks):
        """Test initialisation de tous les frameworks."""
        orchestrator = FrameworkOrchestrator()
        framework1 = MockFramework("framework1")
        framework2 = MockFramework("framework2")
        
        orchestrator.register_framework("test1", framework1)
        orchestrator.register_framework("test2", framework2)
        
        results = await orchestrator.initialize_all()
        
        assert results["status"] == "success"
        assert "test1" in results["frameworks"]
        assert "test2" in results["frameworks"]
        assert framework1.init_called is True
        assert framework2.init_called is True
        
    @pytest.mark.asyncio
    async def test_initialize_all_with_failure(self, clean_frameworks):
        """Test initialisation avec échec partiel."""
        orchestrator = FrameworkOrchestrator()
        framework1 = MockFramework("framework1")
        framework2 = MockFramework("framework2", fail_init=True)
        
        orchestrator.register_framework("test1", framework1)
        orchestrator.register_framework("test2", framework2)
        
        results = await orchestrator.initialize_all()
        
        assert results["status"] == "partial_success"
        assert len(results["failed"]) == 1
        assert "test2" in results["failed"]
        assert framework1.init_called is True
        assert framework2.init_called is True
        
    @pytest.mark.asyncio
    async def test_health_check_all(self, clean_frameworks):
        """Test health check de tous les frameworks."""
        orchestrator = FrameworkOrchestrator()
        framework1 = MockFramework("framework1")
        framework2 = MockFramework("framework2", fail_health=True)
        
        orchestrator.register_framework("test1", framework1)
        orchestrator.register_framework("test2", framework2)
        
        # Initialiser d'abord
        await orchestrator.initialize_all()
        
        health_status = await orchestrator.get_health_status()
        
        assert "test1" in health_status
        assert "test2" in health_status
        assert health_status["test1"].status == FrameworkStatus.RUNNING
        assert health_status["test2"].status == FrameworkStatus.ERROR
        
    @pytest.mark.asyncio
    async def test_shutdown_all_frameworks(self, clean_frameworks):
        """Test arrêt de tous les frameworks."""
        orchestrator = FrameworkOrchestrator()
        framework1 = MockFramework("framework1")
        framework2 = MockFramework("framework2")
        
        orchestrator.register_framework("test1", framework1)
        orchestrator.register_framework("test2", framework2)
        
        # Initialiser et arrêter
        await orchestrator.initialize_all()
        results = await orchestrator.shutdown_all()
        
        assert results["status"] == "success"
        assert framework1.shutdown_called is True
        assert framework2.shutdown_called is True
        assert framework1.status == FrameworkStatus.STOPPED
        assert framework2.status == FrameworkStatus.STOPPED
        
    @pytest.mark.asyncio
    async def test_framework_dependencies(self, clean_frameworks):
        """Test gestion des dépendances entre frameworks."""
        orchestrator = FrameworkOrchestrator()
        
        # Framework avec dépendance
        framework1 = MockFramework("framework1")
        framework2 = MockFramework("framework2")
        framework2.dependencies = ["test1"]  # Dépend de test1
        
        orchestrator.register_framework("test1", framework1)
        orchestrator.register_framework("test2", framework2)
        
        # L'orchestrator devrait initialiser test1 avant test2
        results = await orchestrator.initialize_all()
        
        assert results["status"] == "success"
        assert framework1.init_called is True
        assert framework2.init_called is True
        
    @pytest.mark.asyncio
    async def test_framework_metrics_collection(self, clean_frameworks):
        """Test collecte des métriques des frameworks."""
        orchestrator = FrameworkOrchestrator()
        framework = MockFramework("test_framework")
        framework.metrics = {
            "requests": 100,
            "errors": 5,
            "latency": 0.25
        }
        
        orchestrator.register_framework("test", framework)
        await orchestrator.initialize_framework("test")
        
        metrics = orchestrator.get_metrics()
        
        assert "test" in metrics
        assert metrics["test"]["requests"] == 100
        assert metrics["test"]["errors"] == 5
        assert metrics["test"]["latency"] == 0.25
        
    @pytest.mark.asyncio
    async def test_orchestrator_restart_framework(self, clean_frameworks):
        """Test redémarrage d'un framework."""
        orchestrator = FrameworkOrchestrator()
        framework = MockFramework("test_framework")
        orchestrator.register_framework("test", framework)
        
        # Initialiser
        await orchestrator.initialize_framework("test")
        assert framework.status == FrameworkStatus.RUNNING
        
        # Redémarrer
        result = await orchestrator.restart_framework("test")
        
        assert result is True
        assert framework.shutdown_called is True
        assert framework.status == FrameworkStatus.RUNNING  # Re-initialisé


@pytest.mark.core
@pytest.mark.integration
class TestFrameworkOrchestatorIntegration:
    """Tests d'intégration du Framework Orchestrator."""
    
    @pytest.mark.asyncio
    async def test_full_lifecycle_multiple_frameworks(self, clean_frameworks):
        """Test cycle de vie complet avec plusieurs frameworks."""
        orchestrator = FrameworkOrchestrator()
        
        # Créer plusieurs frameworks avec différentes configurations
        frameworks = {
            "core": MockFramework("core"),
            "security": MockFramework("security"),
            "monitoring": MockFramework("monitoring", fail_health=False),
            "ml": MockFramework("ml")
        }
        
        # Enregistrer tous les frameworks
        for name, framework in frameworks.items():
            orchestrator.register_framework(name, framework)
            
        # Initialisation complète
        init_results = await orchestrator.initialize_all()
        assert init_results["status"] == "success"
        
        # Vérifier health status
        health = await orchestrator.get_health_status()
        for name in frameworks.keys():
            assert health[name].status == FrameworkStatus.RUNNING
            
        # Collecter métriques
        metrics = orchestrator.get_metrics()
        assert len(metrics) == len(frameworks)
        
        # Arrêt graceful
        shutdown_results = await orchestrator.shutdown_all()
        assert shutdown_results["status"] == "success"
        
        # Vérifier que tous sont arrêtés
        for framework in frameworks.values():
            assert framework.shutdown_called is True
            assert framework.status == FrameworkStatus.STOPPED
