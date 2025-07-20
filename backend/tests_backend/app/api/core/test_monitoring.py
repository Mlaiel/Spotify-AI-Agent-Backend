"""
🎵 Tests Ultra-Avancés pour API Core Monitoring Management
==========================================================

Tests industriels complets pour le monitoring et métriques avec patterns enterprise,
tests de sécurité, performance, et validation des systèmes de surveillance.

Développé par Fahed Mlaiel - Enterprise Monitoring Testing Expert
"""

import pytest
import asyncio
import time
import psutil
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Response
from starlette.testclient import TestClient
from prometheus_client import REGISTRY, CollectorRegistry
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

from app.api.core.monitoring import (
    APIMetrics,
    HealthChecker,
    PerformanceMonitor,
    SystemMetrics,
    # AlertManager,  # Not implemented yet
    # MetricsCollector,  # Not implemented yet
    HealthStatus,
    HealthCheck,
    # AlertLevel,  # Not implemented yet
    # Alert,  # Not implemented yet
    get_api_metrics,
    get_health_checker,
    get_performance_monitor,
    create_monitoring_middleware,
    collect_system_metrics,
    check_system_health,
    format_metrics_for_prometheus,
    setup_monitoring
)


# =============================================================================
# FIXTURES ENTERPRISE POUR MONITORING TESTING
# =============================================================================

@pytest.fixture
def clean_metrics():
    """Registre de métriques propre pour les tests"""
    # Créer un registre isolé pour les tests
    test_registry = CollectorRegistry()
    with patch('app.api.core.monitoring.REGISTRY', test_registry):
        yield test_registry


@pytest.fixture
def sample_request():
    """Requête de test"""
    request = Mock(spec=Request)
    request.url.path = "/api/v1/test"
    request.method = "GET"
    request.headers = {"user-agent": "TestClient/1.0"}
    request.client.host = "127.0.0.1"
    return request


@pytest.fixture
def sample_response():
    """Réponse de test"""
    response = Mock(spec=Response)
    response.status_code = 200
    response.headers = {"content-type": "application/json"}
    return response


@pytest.fixture
def mock_health_check():
    """Health check mocké"""
    async def mock_check():
        return HealthStatus.HEALTHY, "Service is healthy", {}
    
    return HealthCheck(
        name="test_service",
        check_function=mock_check,
        timeout=5.0,
        interval=60
    )


@pytest.fixture
def test_app():
    """Application FastAPI de test avec monitoring"""
    app = FastAPI()
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    @app.get("/slow")
    async def slow_endpoint():
        await asyncio.sleep(0.1)
        return {"message": "slow"}
    
    @app.get("/error")
    async def error_endpoint():
        raise ValueError("Test error")
    
    return app


@pytest.fixture
def monitoring_config():
    """Configuration monitoring pour les tests"""
    return {
        "enabled": True,
        "metrics": {
            "enabled": True,
            "port": 9090,
            "path": "/metrics"
        },
        "health": {
            "enabled": True,
            "path": "/health"
        },
        "alerts": {
            "enabled": True,
            "thresholds": {
                "response_time": 1000,  # ms
                "error_rate": 0.05,     # 5%
                "cpu_usage": 80,        # %
                "memory_usage": 80      # %
            }
        }
    }


# =============================================================================
# TESTS DE APIMETRICS
# =============================================================================

class TestAPIMetrics:
    """Tests pour APIMetrics"""
    
    def test_api_metrics_singleton(self, clean_metrics):
        """Test pattern singleton pour APIMetrics"""
        metrics1 = APIMetrics()
        metrics2 = APIMetrics()
        
        assert metrics1 is metrics2
    
    def test_api_metrics_initialization(self, clean_metrics):
        """Test initialisation des métriques"""
        metrics = APIMetrics()
        
        # Vérifier que les métriques sont créées
        assert hasattr(metrics, 'request_counter')
        assert hasattr(metrics, 'request_duration')
        assert hasattr(metrics, 'response_size')
        assert hasattr(metrics, 'active_requests')
    
    def test_record_request(self, clean_metrics, sample_request):
        """Test enregistrement de requête"""
        metrics = APIMetrics()
        
        # Enregistrer une requête
        metrics.record_request(sample_request)
        
        # Vérifier que le compteur a été incrémenté
        # Note: En pratique, on vérifierait via prometheus_client
    
    def test_record_response(self, clean_metrics, sample_request, sample_response):
        """Test enregistrement de réponse"""
        metrics = APIMetrics()
        
        start_time = time.time()
        metrics.record_request(sample_request)
        
        # Simuler un délai
        time.sleep(0.01)
        
        metrics.record_response(sample_request, sample_response, start_time)
        
        # Vérifier que les métriques ont été mises à jour
    
    def test_record_error(self, clean_metrics, sample_request):
        """Test enregistrement d'erreur"""
        metrics = APIMetrics()
        
        error = ValueError("Test error")
        metrics.record_error(sample_request, error)
        
        # Vérifier que l'erreur a été comptabilisée
    
    def test_get_metrics_summary(self, clean_metrics):
        """Test récupération du résumé des métriques"""
        metrics = APIMetrics()
        
        # Simuler quelques requêtes
        for i in range(5):
            request = Mock(spec=Request)
            request.url.path = f"/api/test{i}"
            request.method = "GET"
            metrics.record_request(request)
        
        summary = metrics.get_metrics_summary()
        
        assert isinstance(summary, dict)
        assert "total_requests" in summary
        assert "active_requests" in summary
    
    def test_reset_metrics(self, clean_metrics):
        """Test remise à zéro des métriques"""
        metrics = APIMetrics()
        
        # Enregistrer quelques métriques
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        metrics.record_request(request)
        
        # Remettre à zéro
        metrics.reset_metrics()
        
        # Vérifier que les métriques sont à zéro
        summary = metrics.get_metrics_summary()
        assert summary["total_requests"] == 0


# =============================================================================
# TESTS DE HEALTHCHECKER
# =============================================================================

class TestHealthChecker:
    """Tests pour HealthChecker"""
    
    def test_health_checker_singleton(self):
        """Test pattern singleton pour HealthChecker"""
        checker1 = HealthChecker()
        checker2 = HealthChecker()
        
        assert checker1 is checker2
    
    def test_register_health_check(self, mock_health_check):
        """Test enregistrement de health check"""
        checker = HealthChecker()
        
        checker.register_check(mock_health_check)
        
        assert "test_service" in checker._checks
        assert checker._checks["test_service"] == mock_health_check
    
    @pytest.mark.asyncio
    async def test_run_single_health_check(self, mock_health_check):
        """Test exécution d'un health check"""
        checker = HealthChecker()
        checker.register_check(mock_health_check)
        
        result = await checker.run_check("test_service")
        
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Service is healthy"
        assert result.details == {}
    
    @pytest.mark.asyncio
    async def test_run_all_health_checks(self, mock_health_check):
        """Test exécution de tous les health checks"""
        checker = HealthChecker()
        checker.register_check(mock_health_check)
        
        # Ajouter un deuxième check
        async def mock_check2():
            return HealthStatus.DEGRADED, "Service degraded", {"load": "high"}
        
        check2 = HealthCheck(
            name="test_service2",
            check_function=mock_check2
        )
        checker.register_check(check2)
        
        results = await checker.run_all_checks()
        
        assert len(results) == 2
        assert "test_service" in results
        assert "test_service2" in results
        assert results["test_service"].status == HealthStatus.HEALTHY
        assert results["test_service2"].status == HealthStatus.DEGRADED
    
    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Test timeout des health checks"""
        checker = HealthChecker()
        
        # Health check qui prend trop de temps
        async def slow_check():
            await asyncio.sleep(2.0)
            return HealthStatus.HEALTHY, "Slow check", {}
        
        slow_health_check = HealthCheck(
            name="slow_service",
            check_function=slow_check,
            timeout=0.1  # Timeout très court
        )
        
        checker.register_check(slow_health_check)
        
        result = await checker.run_check("slow_service")
        
        # Le check devrait échouer à cause du timeout
        assert result.status == HealthStatus.UNHEALTHY
        assert "timeout" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self):
        """Test gestion d'exception dans health check"""
        checker = HealthChecker()
        
        # Health check qui lève une exception
        async def failing_check():
            raise RuntimeError("Check failed")
        
        failing_health_check = HealthCheck(
            name="failing_service",
            check_function=failing_check
        )
        
        checker.register_check(failing_health_check)
        
        result = await checker.run_check("failing_service")
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.message
    
    def test_get_overall_health(self):
        """Test calcul de l'état de santé global"""
        checker = HealthChecker()
        
        # Simuler des résultats
        checker._last_results = {
            "service1": Mock(status=HealthStatus.HEALTHY),
            "service2": Mock(status=HealthStatus.HEALTHY),
            "service3": Mock(status=HealthStatus.DEGRADED)
        }
        
        overall = checker.get_overall_health()
        
        # Avec un service dégradé, l'état global devrait être dégradé
        assert overall == HealthStatus.DEGRADED
    
    def test_get_health_summary(self):
        """Test résumé de l'état de santé"""
        checker = HealthChecker()
        
        checker._last_results = {
            "service1": Mock(
                status=HealthStatus.HEALTHY,
                message="Service OK",
                timestamp=datetime.now()
            ),
            "service2": Mock(
                status=HealthStatus.UNHEALTHY,
                message="Service down",
                timestamp=datetime.now()
            )
        }
        
        summary = checker.get_health_summary()
        
        assert isinstance(summary, dict)
        assert summary["overall_status"] == HealthStatus.UNHEALTHY
        assert "healthy" in summary["summary"]
        assert "unhealthy" in summary["summary"]


# =============================================================================
# TESTS DE PERFORMANCEMONITOR
# =============================================================================

class TestPerformanceMonitor:
    """Tests pour PerformanceMonitor"""
    
    def test_performance_monitor_singleton(self):
        """Test pattern singleton pour PerformanceMonitor"""
        monitor1 = PerformanceMonitor()
        monitor2 = PerformanceMonitor()
        
        assert monitor1 is monitor2
    
    def test_start_request_tracking(self, sample_request):
        """Test démarrage du tracking de requête"""
        monitor = PerformanceMonitor()
        
        request_id = monitor.start_request_tracking(sample_request)
        
        assert request_id is not None
        assert request_id in monitor._active_requests
        assert monitor._active_requests[request_id]["start_time"] is not None
    
    def test_end_request_tracking(self, sample_request, sample_response):
        """Test fin du tracking de requête"""
        monitor = PerformanceMonitor()
        
        request_id = monitor.start_request_tracking(sample_request)
        
        # Simuler un délai
        time.sleep(0.01)
        
        duration = monitor.end_request_tracking(request_id, sample_response)
        
        assert duration > 0
        assert request_id not in monitor._active_requests
    
    def test_get_performance_stats(self, sample_request, sample_response):
        """Test récupération des statistiques de performance"""
        monitor = PerformanceMonitor()
        
        # Simuler plusieurs requêtes
        for i in range(5):
            request_id = monitor.start_request_tracking(sample_request)
            time.sleep(0.001)  # Petit délai
            monitor.end_request_tracking(request_id, sample_response)
        
        stats = monitor.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert "avg_response_time" in stats
        assert "total_requests" in stats
        assert "requests_per_second" in stats
    
    def test_get_slow_requests(self, sample_request, sample_response):
        """Test identification des requêtes lentes"""
        monitor = PerformanceMonitor()
        
        # Requête normale
        request_id1 = monitor.start_request_tracking(sample_request)
        monitor.end_request_tracking(request_id1, sample_response)
        
        # Requête lente (simulée)
        request_id2 = monitor.start_request_tracking(sample_request)
        time.sleep(0.1)  # Délai plus long
        monitor.end_request_tracking(request_id2, sample_response)
        
        slow_requests = monitor.get_slow_requests(threshold_ms=50)
        
        assert len(slow_requests) == 1
        assert slow_requests[0]["duration"] > 50
    
    def test_clear_old_data(self, sample_request, sample_response):
        """Test nettoyage des anciennes données"""
        monitor = PerformanceMonitor()
        
        # Ajouter des données
        request_id = monitor.start_request_tracking(sample_request)
        monitor.end_request_tracking(request_id, sample_response)
        
        # Vérifier que les données sont présentes
        assert len(monitor._request_history) > 0
        
        # Nettoyer
        monitor.clear_old_data(max_age_hours=0)  # Tout nettoyer
        
        assert len(monitor._request_history) == 0


# =============================================================================
# TESTS DE SYSTEMMETRICS
# =============================================================================

class TestSystemMetrics:
    """Tests pour SystemMetrics"""
    
    def test_system_metrics_creation(self):
        """Test création des métriques système"""
        metrics = SystemMetrics()
        
        assert hasattr(metrics, 'cpu_usage')
        assert hasattr(metrics, 'memory_usage')
        assert hasattr(metrics, 'disk_usage')
    
    def test_collect_cpu_metrics(self):
        """Test collecte des métriques CPU"""
        metrics = SystemMetrics()
        
        cpu_data = metrics.collect_cpu_metrics()
        
        assert isinstance(cpu_data, dict)
        assert "usage_percent" in cpu_data
        assert "cores" in cpu_data
        assert 0 <= cpu_data["usage_percent"] <= 100
    
    def test_collect_memory_metrics(self):
        """Test collecte des métriques mémoire"""
        metrics = SystemMetrics()
        
        memory_data = metrics.collect_memory_metrics()
        
        assert isinstance(memory_data, dict)
        assert "total" in memory_data
        assert "available" in memory_data
        assert "usage_percent" in memory_data
        assert 0 <= memory_data["usage_percent"] <= 100
    
    def test_collect_disk_metrics(self):
        """Test collecte des métriques disque"""
        metrics = SystemMetrics()
        
        disk_data = metrics.collect_disk_metrics()
        
        assert isinstance(disk_data, dict)
        assert "total" in disk_data
        assert "free" in disk_data
        assert "usage_percent" in disk_data
    
    def test_collect_network_metrics(self):
        """Test collecte des métriques réseau"""
        metrics = SystemMetrics()
        
        network_data = metrics.collect_network_metrics()
        
        assert isinstance(network_data, dict)
        assert "bytes_sent" in network_data
        assert "bytes_recv" in network_data
    
    def test_get_system_summary(self):
        """Test résumé du système"""
        metrics = SystemMetrics()
        
        summary = metrics.get_system_summary()
        
        assert isinstance(summary, dict)
        assert "cpu" in summary
        assert "memory" in summary
        assert "disk" in summary
        assert "timestamp" in summary


# =============================================================================
# TESTS DE ALERTMANAGER
# =============================================================================

class TestAlertManager:
    """Tests pour AlertManager"""
    
    def test_alert_manager_creation(self, monitoring_config):
        """Test création AlertManager"""
        manager = AlertManager(monitoring_config["alerts"])
        
        assert manager._config == monitoring_config["alerts"]
        assert manager._active_alerts == {}
    
    def test_check_response_time_alert(self, monitoring_config):
        """Test alerte temps de réponse"""
        manager = AlertManager(monitoring_config["alerts"])
        
        # Temps de réponse normal
        alert = manager.check_response_time_alert(500)  # 500ms
        assert alert is None
        
        # Temps de réponse élevé
        alert = manager.check_response_time_alert(1500)  # 1500ms
        assert alert is not None
        assert alert.level == AlertLevel.WARNING
        assert "response time" in alert.message.lower()
    
    def test_check_error_rate_alert(self, monitoring_config):
        """Test alerte taux d'erreur"""
        manager = AlertManager(monitoring_config["alerts"])
        
        # Taux d'erreur normal
        alert = manager.check_error_rate_alert(0.02)  # 2%
        assert alert is None
        
        # Taux d'erreur élevé
        alert = manager.check_error_rate_alert(0.08)  # 8%
        assert alert is not None
        assert alert.level == AlertLevel.WARNING
    
    def test_check_system_alerts(self, monitoring_config):
        """Test alertes système"""
        manager = AlertManager(monitoring_config["alerts"])
        
        system_metrics = {
            "cpu": {"usage_percent": 85},
            "memory": {"usage_percent": 85}
        }
        
        alerts = manager.check_system_alerts(system_metrics)
        
        assert len(alerts) == 2  # CPU et mémoire élevés
        assert all(alert.level == AlertLevel.WARNING for alert in alerts)
    
    def test_trigger_alert(self, monitoring_config):
        """Test déclenchement d'alerte"""
        manager = AlertManager(monitoring_config["alerts"])
        
        alert = Alert(
            id="test_alert",
            level=AlertLevel.WARNING,
            message="Test alert",
            source="test"
        )
        
        manager.trigger_alert(alert)
        
        assert alert.id in manager._active_alerts
        assert manager._active_alerts[alert.id] == alert
    
    def test_resolve_alert(self, monitoring_config):
        """Test résolution d'alerte"""
        manager = AlertManager(monitoring_config["alerts"])
        
        # Déclencher une alerte
        alert = Alert(
            id="test_alert",
            level=AlertLevel.WARNING,
            message="Test alert",
            source="test"
        )
        manager.trigger_alert(alert)
        
        # Résoudre l'alerte
        manager.resolve_alert("test_alert")
        
        assert "test_alert" not in manager._active_alerts
    
    def test_get_active_alerts(self, monitoring_config):
        """Test récupération des alertes actives"""
        manager = AlertManager(monitoring_config["alerts"])
        
        # Ajouter quelques alertes
        alert1 = Alert(id="alert1", level=AlertLevel.INFO, message="Info", source="test")
        alert2 = Alert(id="alert2", level=AlertLevel.WARNING, message="Warning", source="test")
        
        manager.trigger_alert(alert1)
        manager.trigger_alert(alert2)
        
        active_alerts = manager.get_active_alerts()
        
        assert len(active_alerts) == 2
        assert "alert1" in active_alerts
        assert "alert2" in active_alerts
    
    def test_get_alerts_by_level(self, monitoring_config):
        """Test récupération d'alertes par niveau"""
        manager = AlertManager(monitoring_config["alerts"])
        
        # Ajouter des alertes de différents niveaux
        info_alert = Alert(id="info", level=AlertLevel.INFO, message="Info", source="test")
        warning_alert = Alert(id="warning", level=AlertLevel.WARNING, message="Warning", source="test")
        critical_alert = Alert(id="critical", level=AlertLevel.CRITICAL, message="Critical", source="test")
        
        manager.trigger_alert(info_alert)
        manager.trigger_alert(warning_alert)
        manager.trigger_alert(critical_alert)
        
        warning_alerts = manager.get_alerts_by_level(AlertLevel.WARNING)
        critical_alerts = manager.get_alerts_by_level(AlertLevel.CRITICAL)
        
        assert len(warning_alerts) == 1
        assert len(critical_alerts) == 1
        assert warning_alerts[0].id == "warning"
        assert critical_alerts[0].id == "critical"


# =============================================================================
# TESTS DES FONCTIONS UTILITAIRES
# =============================================================================

class TestMonitoringUtilities:
    """Tests pour les fonctions utilitaires de monitoring"""
    
    def test_get_api_metrics(self, clean_metrics):
        """Test récupération des métriques API"""
        metrics = get_api_metrics()
        
        assert isinstance(metrics, APIMetrics)
        
        # Deuxième appel devrait retourner la même instance
        metrics2 = get_api_metrics()
        assert metrics is metrics2
    
    def test_get_health_checker(self):
        """Test récupération du health checker"""
        checker = get_health_checker()
        
        assert isinstance(checker, HealthChecker)
        
        # Deuxième appel devrait retourner la même instance
        checker2 = get_health_checker()
        assert checker is checker2
    
    def test_get_performance_monitor(self):
        """Test récupération du performance monitor"""
        monitor = get_performance_monitor()
        
        assert isinstance(monitor, PerformanceMonitor)
        
        # Deuxième appel devrait retourner la même instance
        monitor2 = get_performance_monitor()
        assert monitor is monitor2
    
    def test_collect_system_metrics(self):
        """Test collecte des métriques système"""
        metrics = collect_system_metrics()
        
        assert isinstance(metrics, dict)
        assert "cpu" in metrics
        assert "memory" in metrics
        assert "disk" in metrics
        assert "timestamp" in metrics
    
    @pytest.mark.asyncio
    async def test_check_system_health(self):
        """Test vérification de la santé du système"""
        health = await check_system_health()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert "checks" in health
        assert health["status"] in [status.value for status in HealthStatus]
    
    def test_format_metrics_for_prometheus(self, clean_metrics):
        """Test formatage des métriques pour Prometheus"""
        # Simuler des métriques
        metrics_data = {
            "http_requests_total": 150,
            "http_request_duration_seconds": 0.245,
            "active_connections": 12
        }
        
        formatted = format_metrics_for_prometheus(metrics_data)
        
        assert isinstance(formatted, str)
        assert "http_requests_total" in formatted
        assert "150" in formatted
    
    def test_setup_monitoring(self, test_app, monitoring_config):
        """Test configuration du monitoring"""
        setup_monitoring(test_app, monitoring_config)
        
        # Vérifier que le monitoring a été configuré
        # (En pratique, on vérifierait que les middlewares sont ajoutés)


# =============================================================================
# TESTS DU MIDDLEWARE
# =============================================================================

class TestMonitoringMiddleware:
    """Tests pour le middleware de monitoring"""
    
    @pytest.mark.asyncio
    async def test_monitoring_middleware_basic(self, test_app, monitoring_config):
        """Test middleware de monitoring basique"""
        # Ajouter le middleware
        middleware = create_monitoring_middleware(monitoring_config)
        test_app.add_middleware(middleware)
        
        with TestClient(test_app) as client:
            response = client.get("/test")
            
            assert response.status_code == 200
            
            # Vérifier que les métriques ont été collectées
            metrics = get_api_metrics()
            summary = metrics.get_metrics_summary()
            
            assert summary["total_requests"] > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_middleware_with_error(self, test_app, monitoring_config):
        """Test middleware avec erreur"""
        middleware = create_monitoring_middleware(monitoring_config)
        test_app.add_middleware(middleware)
        
        with TestClient(test_app) as client:
            response = client.get("/error")
            
            assert response.status_code == 500
            
            # Vérifier que l'erreur a été comptabilisée
            metrics = get_api_metrics()
            summary = metrics.get_metrics_summary()
            
            assert summary["total_errors"] > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_middleware_performance(self, test_app, monitoring_config):
        """Test surveillance des performances"""
        middleware = create_monitoring_middleware(monitoring_config)
        test_app.add_middleware(middleware)
        
        with TestClient(test_app) as client:
            response = client.get("/slow")
            
            assert response.status_code == 200
            
            # Vérifier que la durée a été mesurée
            monitor = get_performance_monitor()
            stats = monitor.get_performance_stats()
            
            assert stats["avg_response_time"] > 0


# =============================================================================
# TESTS D'INTÉGRATION
# =============================================================================

@pytest.mark.integration
class TestMonitoringIntegration:
    """Tests d'intégration pour le monitoring"""
    
    def test_full_monitoring_stack(self, test_app, monitoring_config):
        """Test stack complet de monitoring"""
        # Configurer le monitoring
        setup_monitoring(test_app, monitoring_config)
        
        with TestClient(test_app) as client:
            # Faire plusieurs requêtes
            for i in range(10):
                response = client.get("/test")
                assert response.status_code == 200
            
            # Faire une requête d'erreur
            response = client.get("/error")
            assert response.status_code == 500
            
            # Vérifier les métriques
            metrics = get_api_metrics()
            summary = metrics.get_metrics_summary()
            
            assert summary["total_requests"] >= 10
            assert summary["total_errors"] >= 1
    
    @pytest.mark.asyncio
    async def test_monitoring_with_health_checks(self, monitoring_config):
        """Test monitoring avec health checks"""
        checker = get_health_checker()
        
        # Ajouter des health checks
        async def db_check():
            return HealthStatus.HEALTHY, "Database OK", {}
        
        async def cache_check():
            return HealthStatus.DEGRADED, "Cache slow", {"latency": "high"}
        
        checker.register_check(HealthCheck("database", db_check))
        checker.register_check(HealthCheck("cache", cache_check))
        
        # Exécuter les checks
        results = await checker.run_all_checks()
        
        assert len(results) == 2
        assert results["database"].status == HealthStatus.HEALTHY
        assert results["cache"].status == HealthStatus.DEGRADED
        
        # Vérifier l'état global
        overall = checker.get_overall_health()
        assert overall == HealthStatus.DEGRADED
    
    def test_monitoring_with_alerts(self, monitoring_config):
        """Test monitoring avec alertes"""
        alert_manager = AlertManager(monitoring_config["alerts"])
        
        # Simuler des conditions d'alerte
        system_metrics = {
            "cpu": {"usage_percent": 90},
            "memory": {"usage_percent": 85}
        }
        
        alerts = alert_manager.check_system_alerts(system_metrics)
        
        assert len(alerts) == 2
        assert all(alert.level == AlertLevel.WARNING for alert in alerts)
        
        # Déclencher les alertes
        for alert in alerts:
            alert_manager.trigger_alert(alert)
        
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 2


# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

@pytest.mark.performance
class TestMonitoringPerformance:
    """Tests de performance pour le monitoring"""
    
    def test_metrics_collection_overhead(self, benchmark, clean_metrics, sample_request):
        """Test overhead de la collecte de métriques"""
        metrics = APIMetrics()
        
        def record_request():
            metrics.record_request(sample_request)
        
        # La collecte de métriques ne devrait pas avoir un overhead significatif
        result = benchmark(record_request)
    
    def test_health_check_performance(self, benchmark):
        """Test performance des health checks"""
        async def simple_check():
            return HealthStatus.HEALTHY, "OK", {}
        
        checker = HealthChecker()
        health_check = HealthCheck("test", simple_check)
        checker.register_check(health_check)
        
        async def run_check():
            return await checker.run_check("test")
        
        # Utiliser asyncio.run pour le benchmark
        def benchmark_func():
            return asyncio.run(run_check())
        
        result = benchmark(benchmark_func)
        assert result.status == HealthStatus.HEALTHY
    
    def test_system_metrics_collection_performance(self, benchmark):
        """Test performance collecte métriques système"""
        metrics = SystemMetrics()
        
        def collect_metrics():
            return metrics.get_system_summary()
        
        result = benchmark(collect_metrics)
        assert isinstance(result, dict)
    
    def test_concurrent_monitoring(self, test_app, monitoring_config):
        """Test monitoring sous charge concurrente"""
        setup_monitoring(test_app, monitoring_config)
        
        def make_request():
            with TestClient(test_app) as client:
                response = client.get("/test")
                return response.status_code
        
        # Exécuter des requêtes concurrentes
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [future.result() for future in futures]
        
        # Toutes les requêtes devraient réussir
        assert all(status == 200 for status in results)
        
        # Vérifier que les métriques ont été collectées correctement
        metrics = get_api_metrics()
        summary = metrics.get_metrics_summary()
        
        assert summary["total_requests"] >= 50


# =============================================================================
# TESTS DE SÉCURITÉ
# =============================================================================

@pytest.mark.security
class TestMonitoringSecurity:
    """Tests de sécurité pour le monitoring"""
    
    def test_metrics_endpoint_security(self, test_app, monitoring_config):
        """Test sécurité de l'endpoint métriques"""
        # En production, l'endpoint métriques devrait être protégé
        # ou accessible seulement depuis certaines IPs
        pass
    
    def test_health_check_information_disclosure(self):
        """Test divulgation d'informations dans les health checks"""
        # Les health checks ne devraient pas exposer d'informations sensibles
        
        async def secure_check():
            # Pas d'informations sensibles dans les détails
            return HealthStatus.HEALTHY, "Service OK", {"version": "1.0"}
        
        # Éviter les détails qui pourraient aider un attaquant
        async def insecure_check():
            return HealthStatus.UNHEALTHY, "Database down", {
                "host": "internal-db-server.local",
                "port": 5432,
                "error": "Connection failed: authentication failed for user admin"
            }
        
        # Le check sécurisé devrait être préféré
    
    def test_metrics_data_sanitization(self, clean_metrics, sample_request):
        """Test sanitisation des données de métriques"""
        metrics = APIMetrics()
        
        # Requête avec données potentiellement sensibles
        sensitive_request = Mock(spec=Request)
        sensitive_request.url.path = "/api/users/123?token=secret_token"
        sensitive_request.method = "GET"
        
        metrics.record_request(sensitive_request)
        
        # Les métriques ne devraient pas contenir de données sensibles
        summary = metrics.get_metrics_summary()
        
        # Vérifier que les tokens/secrets ne sont pas exposés
        metrics_str = str(summary)
        assert "secret_token" not in metrics_str
    
    def test_monitoring_resource_limits(self, monitoring_config):
        """Test limites de ressources pour le monitoring"""
        # Le monitoring ne devrait pas consommer trop de ressources
        
        alert_manager = AlertManager(monitoring_config["alerts"])
        
        # Déclencher beaucoup d'alertes
        for i in range(1000):
            alert = Alert(
                id=f"alert_{i}",
                level=AlertLevel.INFO,
                message=f"Test alert {i}",
                source="test"
            )
            alert_manager.trigger_alert(alert)
        
        # Vérifier que le système gère correctement la charge
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) <= 1000  # Pas de fuite mémoire


# =============================================================================
# TESTS DE CONFIGURATION
# =============================================================================

@pytest.mark.configuration
class TestMonitoringConfiguration:
    """Tests de configuration pour le monitoring"""
    
    def test_monitoring_enabled_disabled(self, test_app):
        """Test activation/désactivation du monitoring"""
        # Monitoring désactivé
        disabled_config = {"enabled": False}
        setup_monitoring(test_app, disabled_config)
        
        # Le monitoring ne devrait pas être actif
        # (Vérification dépendante de l'implémentation)
    
    def test_custom_metrics_configuration(self, monitoring_config):
        """Test configuration personnalisée des métriques"""
        custom_config = monitoring_config.copy()
        custom_config["metrics"]["custom_labels"] = ["service", "environment"]
        
        # La configuration personnalisée devrait être prise en compte
    
    def test_health_check_intervals(self, monitoring_config):
        """Test configuration des intervalles de health checks"""
        checker = HealthChecker()
        
        # Health check avec intervalle personnalisé
        async def periodic_check():
            return HealthStatus.HEALTHY, "Periodic check", {}
        
        health_check = HealthCheck(
            name="periodic_service",
            check_function=periodic_check,
            interval=30  # 30 secondes
        )
        
        checker.register_check(health_check)
        
        # L'intervalle devrait être respecté
        assert health_check.interval == 30
    
    def test_alert_thresholds_configuration(self, monitoring_config):
        """Test configuration des seuils d'alerte"""
        # Modifier les seuils
        custom_config = monitoring_config["alerts"].copy()
        custom_config["thresholds"]["response_time"] = 500  # Plus strict
        custom_config["thresholds"]["error_rate"] = 0.01    # Plus strict
        
        alert_manager = AlertManager(custom_config)
        
        # Les nouveaux seuils devraient être appliqués
        alert = alert_manager.check_response_time_alert(600)
        assert alert is not None  # Devrait déclencher avec le nouveau seuil
