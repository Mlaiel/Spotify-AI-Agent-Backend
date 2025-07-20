"""
🧪 Tests Monitoring Framework - Observability & Metrics
======================================================

Tests complets du framework de monitoring avec:
- Métriques Prometheus
- Distributed Tracing Jaeger
- Alerting intelligent
- Health monitoring
- System metrics

Développé par: DBA & Data Engineer
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime, timedelta
import prometheus_client

from backend.app.frameworks.monitoring import (
    MonitoringFramework,
    MetricsCollector,
    DistributedTracing,
    AlertManager,
    HealthChecker,
    MonitoringConfig,
    SystemMetrics,
    AlertRule,
    AlertSeverity,
    TraceSpan,
    HealthCheck
)
from backend.app.frameworks import TEST_CONFIG, clean_frameworks, logger


@pytest.fixture
def monitoring_config():
    """Configuration monitoring pour les tests."""
    return MonitoringConfig(
        enable_prometheus=True,
        prometheus_port=TEST_CONFIG["test_metrics_port"],
        enable_tracing=True,
        jaeger_agent_host="localhost",
        jaeger_agent_port=6831,
        enable_alerting=True,
        smtp_server="smtp.test.com",
        smtp_port=587,
        alert_email="admin@test.com",
        cpu_threshold=80.0,
        memory_threshold=85.0,
        disk_threshold=90.0,
        response_time_threshold=2.0,
        error_rate_threshold=5.0
    )


@pytest.fixture
def sample_metrics():
    """Métriques d'exemple pour les tests."""
    return {
        'http_requests_total': 1500,
        'http_request_duration_seconds': 0.25,
        'memory_usage_bytes': 1024 * 1024 * 500,  # 500MB
        'cpu_usage_percent': 65.5,
        'disk_usage_percent': 75.0,
        'active_connections': 50,
        'error_count': 10
    }


@pytest.fixture
def sample_trace_span():
    """Span de trace d'exemple."""
    return TraceSpan(
        trace_id="550e8400-e29b-41d4-a716-446655440000",
        span_id="6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        operation_name="spotify_recommendation",
        start_time=time.time(),
        duration=0.150,
        tags={
            "component": "ml_model",
            "user_id": "user_123",
            "model_version": "v1.0"
        },
        logs=[
            {"timestamp": time.time(), "level": "INFO", "message": "Starting prediction"},
            {"timestamp": time.time() + 0.1, "level": "INFO", "message": "Prediction completed"}
        ]
    )


@pytest.mark.monitoring
class TestMonitoringConfig:
    """Tests de la configuration monitoring."""
    
    def test_monitoring_config_creation(self):
        """Test création configuration monitoring."""
        config = MonitoringConfig(
            enable_prometheus=True,
            prometheus_port=9090
        )
        
        assert config.enable_prometheus is True
        assert config.prometheus_port == 9090
        assert config.enable_tracing is True
        assert config.enable_alerting is True
        assert config.cpu_threshold == 90.0
        
    def test_monitoring_config_validation(self):
        """Test validation configuration."""
        # Port invalide
        with pytest.raises(ValueError, match="Invalid Prometheus port"):
            MonitoringConfig(prometheus_port=0)
            
        # Seuil invalide
        with pytest.raises(ValueError, match="CPU threshold must be between 0 and 100"):
            MonitoringConfig(cpu_threshold=150.0)
            
    def test_monitoring_config_alert_settings(self):
        """Test configuration alertes."""
        config = MonitoringConfig(
            enable_alerting=True,
            smtp_server="mail.example.com",
            smtp_port=587,
            alert_email="alerts@example.com"
        )
        
        assert config.smtp_server == "mail.example.com"
        assert config.smtp_port == 587
        assert config.alert_email == "alerts@example.com"
        
    def test_monitoring_config_thresholds(self):
        """Test seuils de monitoring."""
        config = MonitoringConfig(
            cpu_threshold=75.0,
            memory_threshold=80.0,
            disk_threshold=85.0,
            response_time_threshold=1.5,
            error_rate_threshold=3.0
        )
        
        assert config.cpu_threshold == 75.0
        assert config.memory_threshold == 80.0
        assert config.disk_threshold == 85.0
        assert config.response_time_threshold == 1.5
        assert config.error_rate_threshold == 3.0


@pytest.mark.monitoring
class TestMetricsCollector:
    """Tests du collecteur de métriques."""
    
    def test_metrics_collector_creation(self, monitoring_config):
        """Test création collecteur métriques."""
        collector = MetricsCollector(monitoring_config)
        
        assert collector.prometheus_port == monitoring_config.prometheus_port
        assert collector.metrics == {}
        assert collector.custom_metrics == {}
        
    def test_record_http_request(self, monitoring_config):
        """Test enregistrement requête HTTP."""
        collector = MetricsCollector(monitoring_config)
        
        # Enregistrer requête réussie
        collector.record_http_request(
            method="GET",
            endpoint="/api/recommendations",
            status_code=200,
            duration=0.25
        )
        
        # Vérifier métriques
        assert "http_requests_total" in collector.metrics
        assert "http_request_duration_seconds" in collector.metrics
        
        # Enregistrer requête d'erreur
        collector.record_http_request(
            method="POST",
            endpoint="/api/tracks",
            status_code=500,
            duration=1.5
        )
        
        # Vérifier comptage erreurs
        assert collector.metrics["http_requests_total"]["500"] > 0
        
    def test_record_ai_prediction(self, monitoring_config):
        """Test enregistrement prédiction IA."""
        collector = MetricsCollector(monitoring_config)
        
        collector.record_ai_prediction(
            model_name="spotify_recommendation",
            prediction_type="recommendation",
            duration=0.180,
            confidence=0.95,
            success=True
        )
        
        # Vérifier métriques IA
        assert "ai_predictions_total" in collector.metrics
        assert "ai_prediction_duration_seconds" in collector.metrics
        assert "ai_prediction_confidence" in collector.metrics
        
    def test_record_system_metrics(self, monitoring_config, sample_metrics):
        """Test enregistrement métriques système."""
        collector = MetricsCollector(monitoring_config)
        
        collector.record_system_metrics(
            cpu_usage=sample_metrics['cpu_usage_percent'],
            memory_usage=sample_metrics['memory_usage_bytes'],
            disk_usage=sample_metrics['disk_usage_percent'],
            network_io={"bytes_sent": 1024, "bytes_recv": 2048}
        )
        
        # Vérifier métriques système
        assert "system_cpu_usage_percent" in collector.metrics
        assert "system_memory_usage_bytes" in collector.metrics
        assert "system_disk_usage_percent" in collector.metrics
        assert "system_network_bytes_sent" in collector.metrics
        
    def test_record_spotify_api_metrics(self, monitoring_config):
        """Test métriques API Spotify."""
        collector = MetricsCollector(monitoring_config)
        
        collector.record_spotify_api_call(
            endpoint="tracks",
            status_code=200,
            duration=0.50,
            rate_limit_remaining=999
        )
        
        # Vérifier métriques Spotify
        assert "spotify_api_calls_total" in collector.metrics
        assert "spotify_api_duration_seconds" in collector.metrics
        assert "spotify_api_rate_limit_remaining" in collector.metrics
        
    def test_custom_metrics(self, monitoring_config):
        """Test métriques personnalisées."""
        collector = MetricsCollector(monitoring_config)
        
        # Créer métrique personnalisée
        collector.create_custom_metric(
            name="user_playlists_created",
            metric_type="counter",
            description="Number of playlists created by users"
        )
        
        # Incrémenter métrique
        collector.increment_custom_metric("user_playlists_created", 1, {"user_type": "premium"})
        
        assert "user_playlists_created" in collector.custom_metrics
        
    def test_get_prometheus_metrics(self, monitoring_config):
        """Test export métriques Prometheus."""
        collector = MetricsCollector(monitoring_config)
        
        # Enregistrer quelques métriques
        collector.record_http_request("GET", "/api/test", 200, 0.1)
        collector.record_system_metrics(50.0, 1024*1024*100, 60.0)
        
        # Export Prometheus
        prometheus_output = collector.get_prometheus_metrics()
        
        assert isinstance(prometheus_output, str)
        assert "# HELP" in prometheus_output
        assert "# TYPE" in prometheus_output
        assert "http_requests_total" in prometheus_output


@pytest.mark.monitoring
class TestDistributedTracing:
    """Tests du tracing distribué."""
    
    def test_distributed_tracing_creation(self, monitoring_config):
        """Test création tracing distribué."""
        tracing = DistributedTracing(monitoring_config)
        
        assert tracing.jaeger_agent_host == monitoring_config.jaeger_agent_host
        assert tracing.jaeger_agent_port == monitoring_config.jaeger_agent_port
        assert tracing.active_spans == {}
        
    def test_start_span(self, monitoring_config):
        """Test démarrage span."""
        tracing = DistributedTracing(monitoring_config)
        
        span = tracing.start_span(
            operation_name="user_authentication",
            parent_span_id=None,
            tags={"user_id": "user_123", "method": "jwt"}
        )
        
        assert span is not None
        assert span.operation_name == "user_authentication"
        assert span.tags["user_id"] == "user_123"
        assert span.span_id in tracing.active_spans
        
    def test_finish_span(self, monitoring_config):
        """Test finalisation span."""
        tracing = DistributedTracing(monitoring_config)
        
        span = tracing.start_span("test_operation")
        span_id = span.span_id
        
        # Ajouter logs
        tracing.log_span_event(span_id, "info", "Operation in progress")
        
        # Finaliser span
        finished_span = tracing.finish_span(span_id)
        
        assert finished_span is not None
        assert finished_span.duration > 0
        assert len(finished_span.logs) > 0
        assert span_id not in tracing.active_spans
        
    def test_nested_spans(self, monitoring_config):
        """Test spans imbriqués."""
        tracing = DistributedTracing(monitoring_config)
        
        # Span parent
        parent_span = tracing.start_span("recommendation_pipeline")
        
        # Span enfant
        child_span = tracing.start_span(
            "load_user_data",
            parent_span_id=parent_span.span_id
        )
        
        assert child_span.parent_span_id == parent_span.span_id
        
        # Finaliser dans l'ordre
        tracing.finish_span(child_span.span_id)
        tracing.finish_span(parent_span.span_id)
        
    def test_trace_context_propagation(self, monitoring_config):
        """Test propagation contexte de trace."""
        tracing = DistributedTracing(monitoring_config)
        
        # Créer trace
        parent_span = tracing.start_span("api_request")
        trace_id = parent_span.trace_id
        
        # Extraire contexte
        trace_context = tracing.extract_trace_context(trace_id)
        
        assert trace_context is not None
        assert trace_context["trace_id"] == trace_id
        
        # Injecter contexte dans span enfant
        child_span = tracing.start_span(
            "database_query",
            trace_context=trace_context
        )
        
        assert child_span.trace_id == trace_id
        
    def test_span_logging(self, monitoring_config):
        """Test logging dans spans."""
        tracing = DistributedTracing(monitoring_config)
        
        span = tracing.start_span("ml_prediction")
        span_id = span.span_id
        
        # Ajouter plusieurs logs
        tracing.log_span_event(span_id, "info", "Loading model")
        tracing.log_span_event(span_id, "debug", "Model loaded successfully")
        tracing.log_span_event(span_id, "info", "Running prediction")
        
        finished_span = tracing.finish_span(span_id)
        
        assert len(finished_span.logs) == 3
        assert any("Loading model" in log["message"] for log in finished_span.logs)
        
    def test_export_traces_jaeger(self, monitoring_config):
        """Test export traces vers Jaeger."""
        tracing = DistributedTracing(monitoring_config)
        
        # Créer et finaliser span
        span = tracing.start_span("test_export")
        tracing.finish_span(span.span_id)
        
        # Mock export Jaeger
        with patch.object(tracing, '_send_to_jaeger') as mock_send:
            tracing.export_traces()
            mock_send.assert_called()


@pytest.mark.monitoring
class TestAlertManager:
    """Tests du gestionnaire d'alertes."""
    
    def test_alert_manager_creation(self, monitoring_config):
        """Test création gestionnaire alertes."""
        alert_manager = AlertManager(monitoring_config)
        
        assert alert_manager.smtp_server == monitoring_config.smtp_server
        assert alert_manager.alert_email == monitoring_config.alert_email
        assert alert_manager.alert_rules == []
        assert alert_manager.active_alerts == {}
        
    def test_create_alert_rule(self, monitoring_config):
        """Test création règle d'alerte."""
        alert_manager = AlertManager(monitoring_config)
        
        rule = alert_manager.create_alert_rule(
            name="high_cpu_usage",
            condition="cpu_usage > 80",
            severity=AlertSeverity.WARNING,
            description="CPU usage is too high",
            threshold=80.0
        )
        
        assert rule.name == "high_cpu_usage"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.threshold == 80.0
        assert rule in alert_manager.alert_rules
        
    @pytest.mark.asyncio
    async def test_evaluate_alert_rules(self, monitoring_config, sample_metrics):
        """Test évaluation règles d'alerte."""
        alert_manager = AlertManager(monitoring_config)
        
        # Créer règles
        alert_manager.create_alert_rule(
            "high_cpu", "cpu_usage > 80", AlertSeverity.WARNING, threshold=80.0
        )
        alert_manager.create_alert_rule(
            "low_disk", "disk_usage > 90", AlertSeverity.CRITICAL, threshold=90.0
        )
        
        # Évaluer avec métriques (CPU OK, disk OK)
        triggered_alerts = await alert_manager.evaluate_rules(sample_metrics)
        
        # Aucune alerte ne devrait être déclenchée
        assert len(triggered_alerts) == 0
        
        # Modifier métriques pour déclencher alertes
        high_metrics = sample_metrics.copy()
        high_metrics['cpu_usage_percent'] = 85.0  # > 80
        high_metrics['disk_usage_percent'] = 95.0  # > 90
        
        triggered_alerts = await alert_manager.evaluate_rules(high_metrics)
        
        # Deux alertes devraient être déclenchées
        assert len(triggered_alerts) == 2
        
    @pytest.mark.asyncio
    async def test_send_email_alert(self, monitoring_config):
        """Test envoi alerte par email."""
        alert_manager = AlertManager(monitoring_config)
        
        alert = {
            "name": "test_alert",
            "severity": AlertSeverity.WARNING,
            "message": "Test alert message",
            "timestamp": datetime.utcnow(),
            "metrics": {"cpu_usage": 85.0}
        }
        
        # Mock SMTP
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value = mock_server
            
            await alert_manager.send_email_alert(alert)
            
            # Vérifier appels SMTP
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once()
            mock_server.send_message.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_send_slack_alert(self, monitoring_config):
        """Test envoi alerte Slack."""
        alert_manager = AlertManager(monitoring_config)
        alert_manager.slack_webhook_url = "https://hooks.slack.com/test"
        
        alert = {
            "name": "slack_test_alert",
            "severity": AlertSeverity.CRITICAL,
            "message": "Critical system alert",
            "timestamp": datetime.utcnow()
        }
        
        # Mock requête HTTP
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value.status_code = 200
            
            await alert_manager.send_slack_alert(alert)
            
            mock_post.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_alert_deduplication(self, monitoring_config):
        """Test déduplication alertes."""
        alert_manager = AlertManager(monitoring_config)
        
        # Créer règle
        alert_manager.create_alert_rule(
            "test_rule", "value > 10", AlertSeverity.INFO, threshold=10.0
        )
        
        # Déclencher alerte première fois
        metrics = {"value": 15.0}
        alerts1 = await alert_manager.evaluate_rules(metrics)
        assert len(alerts1) == 1
        
        # Déclencher la même alerte immédiatement
        alerts2 = await alert_manager.evaluate_rules(metrics)
        
        # Ne devrait pas être re-déclenchée (déduplication)
        assert len(alerts2) == 0
        
    @pytest.mark.asyncio
    async def test_alert_recovery(self, monitoring_config):
        """Test récupération alertes."""
        alert_manager = AlertManager(monitoring_config)
        
        # Créer règle
        rule = alert_manager.create_alert_rule(
            "recovery_test", "value > 10", AlertSeverity.WARNING, threshold=10.0
        )
        
        # Déclencher alerte
        high_metrics = {"value": 15.0}
        alerts = await alert_manager.evaluate_rules(high_metrics)
        assert len(alerts) == 1
        
        # Métriques reviennent à la normale
        normal_metrics = {"value": 5.0}
        recovery_alerts = await alert_manager.evaluate_rules(normal_metrics)
        
        # Alerte de récupération devrait être générée
        assert len(recovery_alerts) == 1
        assert recovery_alerts[0]["type"] == "recovery"


@pytest.mark.monitoring
class TestHealthChecker:
    """Tests du vérificateur de santé."""
    
    @pytest.mark.asyncio
    async def test_health_checker_creation(self, monitoring_config):
        """Test création vérificateur santé."""
        health_checker = HealthChecker(monitoring_config)
        
        assert health_checker.config == monitoring_config
        assert health_checker.health_checks == []
        
    @pytest.mark.asyncio
    async def test_register_health_check(self, monitoring_config):
        """Test enregistrement health check."""
        health_checker = HealthChecker(monitoring_config)
        
        async def database_check():
            return HealthCheck(
                name="database",
                status="healthy",
                response_time=0.05,
                details={"connections": 10}
            )
            
        health_checker.register_health_check("database", database_check)
        
        assert len(health_checker.health_checks) == 1
        assert "database" in health_checker.health_checks[0]
        
    @pytest.mark.asyncio
    async def test_run_health_checks(self, monitoring_config):
        """Test exécution health checks."""
        health_checker = HealthChecker(monitoring_config)
        
        # Health check qui réussit
        async def healthy_service():
            return HealthCheck(
                name="healthy_service",
                status="healthy",
                response_time=0.1
            )
            
        # Health check qui échoue
        async def unhealthy_service():
            return HealthCheck(
                name="unhealthy_service",
                status="unhealthy",
                response_time=5.0,
                error="Connection timeout"
            )
            
        health_checker.register_health_check("service1", healthy_service)
        health_checker.register_health_check("service2", unhealthy_service)
        
        # Exécuter tous les checks
        results = await health_checker.run_all_checks()
        
        assert len(results) == 2
        assert results["service1"]["status"] == "healthy"
        assert results["service2"]["status"] == "unhealthy"
        
    @pytest.mark.asyncio
    async def test_system_health_overview(self, monitoring_config):
        """Test overview santé système."""
        health_checker = HealthChecker(monitoring_config)
        
        # Mock métriques système
        with patch.object(health_checker, '_get_system_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "cpu_usage": 45.0,
                "memory_usage": 60.0,
                "disk_usage": 70.0,
                "network_latency": 0.05
            }
            
            overview = await health_checker.get_system_overview()
            
        assert overview["overall_status"] == "healthy"
        assert overview["cpu"]["status"] == "healthy"
        assert overview["memory"]["status"] == "healthy"
        assert overview["disk"]["status"] == "healthy"
        
    @pytest.mark.asyncio
    async def test_dependency_health_checks(self, monitoring_config):
        """Test health checks dépendances."""
        health_checker = HealthChecker(monitoring_config)
        
        # Mock checks dépendances externes
        async def redis_check():
            return HealthCheck("redis", "healthy", 0.01)
            
        async def spotify_api_check():
            return HealthCheck("spotify_api", "healthy", 0.2)
            
        async def database_check():
            return HealthCheck("database", "unhealthy", 5.0, error="Timeout")
            
        health_checker.register_health_check("redis", redis_check)
        health_checker.register_health_check("spotify_api", spotify_api_check)
        health_checker.register_health_check("database", database_check)
        
        results = await health_checker.run_all_checks()
        
        # Calculer santé globale
        overall_health = health_checker.calculate_overall_health(results)
        
        # Une dépendance critique échoue -> santé dégradée
        assert overall_health["status"] == "degraded"
        assert overall_health["healthy_services"] == 2
        assert overall_health["total_services"] == 3


@pytest.mark.monitoring
class TestMonitoringFramework:
    """Tests du framework de monitoring complet."""
    
    @pytest.mark.asyncio
    async def test_monitoring_framework_initialization(self, monitoring_config, clean_frameworks):
        """Test initialisation framework monitoring."""
        monitoring_framework = MonitoringFramework(monitoring_config)
        
        result = await monitoring_framework.initialize()
        
        assert result is True
        assert monitoring_framework.status.name == "RUNNING"
        assert monitoring_framework.metrics_collector is not None
        assert monitoring_framework.distributed_tracing is not None
        assert monitoring_framework.alert_manager is not None
        assert monitoring_framework.health_checker is not None
        
    @pytest.mark.asyncio
    async def test_monitoring_framework_metrics_collection(self, monitoring_config, clean_frameworks):
        """Test collecte métriques framework."""
        monitoring_framework = MonitoringFramework(monitoring_config)
        await monitoring_framework.initialize()
        
        # Enregistrer métriques via framework
        await monitoring_framework.record_http_request("GET", "/api/test", 200, 0.1)
        await monitoring_framework.record_ai_prediction("test_model", "classification", 0.2, 0.9)
        
        # Collecter métriques
        metrics = monitoring_framework.get_metrics()
        
        assert "http_requests_total" in metrics
        assert "ai_predictions_total" in metrics
        
    @pytest.mark.asyncio
    async def test_monitoring_framework_tracing(self, monitoring_config, clean_frameworks):
        """Test tracing framework."""
        monitoring_framework = MonitoringFramework(monitoring_config)
        await monitoring_framework.initialize()
        
        # Démarrer trace
        span = await monitoring_framework.start_trace(
            "test_operation",
            tags={"component": "test"}
        )
        
        assert span is not None
        
        # Ajouter événement
        await monitoring_framework.log_trace_event(span.span_id, "info", "Test event")
        
        # Finaliser trace
        finished_span = await monitoring_framework.finish_trace(span.span_id)
        assert finished_span.duration > 0
        
    @pytest.mark.asyncio
    async def test_monitoring_framework_alerting(self, monitoring_config, clean_frameworks):
        """Test alerting framework."""
        monitoring_framework = MonitoringFramework(monitoring_config)
        await monitoring_framework.initialize()
        
        # Ajouter règle d'alerte
        await monitoring_framework.add_alert_rule(
            "test_alert",
            "test_metric > 100",
            AlertSeverity.WARNING,
            threshold=100.0
        )
        
        # Simuler condition d'alerte
        with patch.object(monitoring_framework.alert_manager, 'evaluate_rules') as mock_eval:
            mock_eval.return_value = [{"name": "test_alert", "triggered": True}]
            
            alerts = await monitoring_framework.check_alerts({"test_metric": 150})
            
        assert len(alerts) == 1
        
    @pytest.mark.asyncio
    async def test_monitoring_framework_health_check(self, monitoring_config, clean_frameworks):
        """Test health check framework."""
        monitoring_framework = MonitoringFramework(monitoring_config)
        await monitoring_framework.initialize()
        
        health = await monitoring_framework.health_check()
        
        assert health.status.name == "RUNNING"
        assert "Monitoring framework" in health.message
        assert "metrics_collector" in health.details
        assert "distributed_tracing" in health.details


@pytest.mark.monitoring
@pytest.mark.integration
class TestMonitoringFrameworkIntegration:
    """Tests d'intégration framework monitoring."""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_pipeline(self, monitoring_config, clean_frameworks):
        """Test pipeline monitoring complet."""
        monitoring_framework = MonitoringFramework(monitoring_config)
        await monitoring_framework.initialize()
        
        # 1. Démarrer trace pour une opération
        span = await monitoring_framework.start_trace(
            "user_recommendation_flow",
            tags={"user_id": "user_123", "session_id": "session_456"}
        )
        
        # 2. Enregistrer métriques pendant l'opération
        await monitoring_framework.record_http_request("POST", "/api/recommendations", 200, 0.5)
        await monitoring_framework.record_ai_prediction("spotify_model", "recommendation", 0.3, 0.87)
        
        # 3. Logger événements dans la trace
        await monitoring_framework.log_trace_event(span.span_id, "info", "Loading user preferences")
        await monitoring_framework.log_trace_event(span.span_id, "info", "Generating recommendations")
        
        # 4. Finaliser trace
        finished_span = await monitoring_framework.finish_trace(span.span_id)
        
        # 5. Vérifier health check global
        health = await monitoring_framework.get_system_health()
        
        # Vérifications
        assert finished_span.duration > 0
        assert len(finished_span.logs) == 2
        assert health["overall_status"] in ["healthy", "degraded"]
        
        # 6. Exporter métriques Prometheus
        prometheus_metrics = monitoring_framework.get_prometheus_metrics()
        assert "http_requests_total" in prometheus_metrics
        assert "ai_predictions_total" in prometheus_metrics
        
    @pytest.mark.asyncio
    async def test_monitoring_with_alerts(self, monitoring_config, clean_frameworks):
        """Test monitoring avec déclenchement alertes."""
        monitoring_framework = MonitoringFramework(monitoring_config)
        await monitoring_framework.initialize()
        
        # Configurer alerte pour taux d'erreur élevé
        await monitoring_framework.add_alert_rule(
            "high_error_rate",
            "error_rate > 5.0",
            AlertSeverity.CRITICAL,
            threshold=5.0
        )
        
        # Simuler beaucoup d'erreurs
        for _ in range(10):
            await monitoring_framework.record_http_request("GET", "/api/test", 500, 0.1)
            
        # Une requête réussie
        await monitoring_framework.record_http_request("GET", "/api/test", 200, 0.1)
        
        # Calculer taux d'erreur et vérifier alertes
        metrics = monitoring_framework.get_metrics()
        error_rate = (metrics.get("http_requests_total", {}).get("500", 0) / 
                     sum(metrics.get("http_requests_total", {}).values())) * 100
        
        # Si taux d'erreur > 5%, alerte devrait être déclenchée
        if error_rate > 5.0:
            with patch.object(monitoring_framework.alert_manager, 'send_email_alert') as mock_email:
                alerts = await monitoring_framework.check_alerts({"error_rate": error_rate})
                if alerts:
                    mock_email.assert_called()


@pytest.mark.monitoring
@pytest.mark.performance
class TestMonitoringFrameworkPerformance:
    """Tests de performance framework monitoring."""
    
    @pytest.mark.asyncio
    async def test_high_volume_metrics_collection(self, monitoring_config, clean_frameworks):
        """Test collecte métriques haut volume."""
        monitoring_framework = MonitoringFramework(monitoring_config)
        await monitoring_framework.initialize()
        
        # Simuler gros volume de métriques
        start_time = time.time()
        
        tasks = []
        for i in range(1000):
            task = monitoring_framework.record_http_request(
                "GET", f"/api/endpoint_{i % 10}", 200, 0.1 + (i % 5) * 0.01
            )
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Collecte devrait être rapide même avec 1000 métriques
        assert duration < 5.0  # Moins de 5 secondes
        
        # Vérifier que toutes les métriques sont enregistrées
        metrics = monitoring_framework.get_metrics()
        assert metrics["http_requests_total"]["200"] == 1000
        
    @pytest.mark.asyncio
    async def test_concurrent_tracing(self, monitoring_config, clean_frameworks):
        """Test tracing concurrent."""
        monitoring_framework = MonitoringFramework(monitoring_config)
        await monitoring_framework.initialize()
        
        async def create_trace(operation_id):
            span = await monitoring_framework.start_trace(f"operation_{operation_id}")
            await asyncio.sleep(0.01)  # Simuler travail
            await monitoring_framework.log_trace_event(span.span_id, "info", f"Op {operation_id}")
            return await monitoring_framework.finish_trace(span.span_id)
            
        # Créer 50 traces concurrentes
        tasks = [create_trace(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        # Toutes les traces doivent être complètes
        assert len(results) == 50
        assert all(span.duration > 0 for span in results)
        assert all(len(span.logs) == 1 for span in results)
