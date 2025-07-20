#!/usr/bin/env python3
"""
Spotify AI Agent - Metrics Exporters Test Suite
===============================================

Suite de tests complète pour tous les exportateurs de métriques
avec tests unitaires, d'intégration et de performance.

Author: Fahed Mlaiel & Spotify AI Team
Date: 2024
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional

# Imports des modules à tester
from .prometheus_exporter import PrometheusMultiTenantExporter
from .grafana_exporter import GrafanaMultiTenantExporter
from .elastic_exporter import ElasticsearchMetricsExporter
from .influxdb_exporter import InfluxDBMetricsExporter
from .custom_exporter import DatadogExporter, NewRelicExporter, SplunkExporter
from .batch_exporter import BatchMetricsExporter
from .streaming_exporter import StreamingMetricsExporter
from .config import ConfigurationManager, ExporterConfiguration, Environment
from .example_usage import SpotifyAIMetrics, SpotifyAIMetricsGenerator, MetricsExportManager

import structlog

# Configuration du logging pour les tests
logger = structlog.get_logger(__name__)


class TestSpotifyAIMetrics:
    """Tests pour la classe SpotifyAIMetrics."""
    
    def test_metrics_creation(self):
        """Test de création d'une métrique."""
        metrics = SpotifyAIMetrics(
            tenant_id="test_tenant",
            timestamp=datetime.now(timezone.utc),
            ai_inference_requests=100,
            ai_model_accuracy=0.95
        )
        
        assert metrics.tenant_id == "test_tenant"
        assert metrics.ai_inference_requests == 100
        assert metrics.ai_model_accuracy == 0.95
        
    def test_metrics_to_dict(self):
        """Test de conversion en dictionnaire."""
        now = datetime.now(timezone.utc)
        metrics = SpotifyAIMetrics(
            tenant_id="test_tenant",
            timestamp=now,
            ai_inference_requests=100
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["tenant_id"] == "test_tenant"
        assert metrics_dict["timestamp"] == now.isoformat()
        assert metrics_dict["ai_inference_requests"] == 100
        
    def test_metrics_to_prometheus_format(self):
        """Test de conversion au format Prometheus."""
        metrics = SpotifyAIMetrics(
            tenant_id="test_tenant",
            timestamp=datetime.now(timezone.utc),
            ai_inference_requests=100,
            ai_model_accuracy=0.95
        )
        
        prometheus_metrics = metrics.to_prometheus_format()
        
        assert len(prometheus_metrics) > 0
        
        # Vérifier la présence de métriques counter
        counter_metrics = [m for m in prometheus_metrics if m["type"] == "counter"]
        assert len(counter_metrics) > 0
        
        # Vérifier la présence de métriques gauge
        gauge_metrics = [m for m in prometheus_metrics if m["type"] == "gauge"]
        assert len(gauge_metrics) > 0
        
        # Vérifier les labels
        for metric in prometheus_metrics:
            assert "tenant_id" in metric["labels"]
            assert metric["labels"]["tenant_id"] == "test_tenant"


class TestSpotifyAIMetricsGenerator:
    """Tests pour le générateur de métriques."""
    
    def test_generator_creation(self):
        """Test de création du générateur."""
        generator = SpotifyAIMetricsGenerator("test_tenant")
        assert generator.tenant_id == "test_tenant"
        assert len(generator.baseline_values) > 0
        
    def test_generate_realistic_metrics(self):
        """Test de génération de métriques réalistes."""
        generator = SpotifyAIMetricsGenerator("test_tenant")
        metrics = generator.generate_realistic_metrics()
        
        assert metrics.tenant_id == "test_tenant"
        assert metrics.ai_inference_requests > 0
        assert 0 <= metrics.ai_model_accuracy <= 1
        assert 0 <= metrics.cache_hit_rate <= 1
        assert metrics.error_rate >= 0
        
    def test_metrics_variance(self):
        """Test de la variance dans les métriques générées."""
        generator = SpotifyAIMetricsGenerator("test_tenant")
        
        metrics_list = []
        for _ in range(10):
            metrics_list.append(generator.generate_realistic_metrics())
            
        # Vérifier que les valeurs varient
        inference_requests = [m.ai_inference_requests for m in metrics_list]
        assert len(set(inference_requests)) > 1  # Au moins 2 valeurs différentes


class TestConfigurationManager:
    """Tests pour le gestionnaire de configuration."""
    
    def test_create_default_configuration(self):
        """Test de création de configuration par défaut."""
        config_manager = ConfigurationManager()
        config = config_manager._create_default_configuration("test_tenant")
        
        assert config.tenant_id == "test_tenant"
        assert config.environment in Environment
        assert config.service_name == "spotify-ai-agent"
        
    def test_configuration_validation(self):
        """Test de validation de configuration."""
        config_manager = ConfigurationManager()
        
        # Configuration valide
        valid_config = config_manager._create_default_configuration("test_tenant")
        errors = config_manager.validate_configuration(valid_config)
        assert len(errors) == 0
        
        # Configuration invalide
        invalid_config = config_manager._create_default_configuration("")  # tenant_id vide
        errors = config_manager.validate_configuration(invalid_config)
        assert len(errors) > 0
        
    def test_environment_templates(self):
        """Test des templates d'environnement."""
        config_manager = ConfigurationManager()
        
        # Test template développement
        dev_template = config_manager.get_environment_template(Environment.DEVELOPMENT)
        assert dev_template["prometheus"]["rate_limit_per_second"] == 100
        
        # Test template production
        prod_template = config_manager.get_environment_template(Environment.PRODUCTION)
        assert prod_template["prometheus"]["rate_limit_per_second"] == 2000
        
    def test_config_to_dict_and_back(self):
        """Test de conversion configuration <-> dictionnaire."""
        config_manager = ConfigurationManager()
        original_config = config_manager._create_default_configuration("test_tenant")
        
        # Convertir en dict puis reconvertir
        config_dict = config_manager._config_to_dict(original_config)
        restored_config = config_manager._dict_to_config(config_dict)
        
        assert original_config.tenant_id == restored_config.tenant_id
        assert original_config.environment == restored_config.environment


@pytest.mark.asyncio
class TestPrometheusExporter:
    """Tests pour l'exportateur Prometheus."""
    
    async def test_exporter_creation(self):
        """Test de création de l'exportateur."""
        exporter = PrometheusMultiTenantExporter(
            tenant_id="test_tenant",
            prometheus_url="http://localhost:9090"
        )
        
        assert exporter.tenant_id == "test_tenant"
        assert exporter.prometheus_url == "http://localhost:9090"
        
    @patch('aiohttp.ClientSession.post')
    async def test_export_metrics(self, mock_post):
        """Test d'export de métriques."""
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.text = AsyncMock(return_value="ok")
        
        exporter = PrometheusMultiTenantExporter(
            tenant_id="test_tenant",
            prometheus_url="http://localhost:9090"
        )
        
        metrics = [
            {
                "name": "test_metric",
                "type": "gauge",
                "value": 42.0,
                "labels": {"tenant_id": "test_tenant"},
                "timestamp": datetime.now(timezone.utc)
            }
        ]
        
        await exporter.export_metrics(metrics)
        mock_post.assert_called()
        
    @patch('aiohttp.ClientSession.get')
    async def test_health_check(self, mock_get):
        """Test de vérification de santé."""
        mock_get.return_value.__aenter__.return_value.status = 200
        
        exporter = PrometheusMultiTenantExporter(
            tenant_id="test_tenant",
            prometheus_url="http://localhost:9090"
        )
        
        health = await exporter.health_check()
        assert health["status"] == "healthy"


@pytest.mark.asyncio
class TestGrafanaExporter:
    """Tests pour l'exportateur Grafana."""
    
    async def test_exporter_creation(self):
        """Test de création de l'exportateur."""
        exporter = GrafanaMultiTenantExporter(
            tenant_id="test_tenant",
            grafana_url="http://localhost:3000",
            api_key="test_key"
        )
        
        assert exporter.tenant_id == "test_tenant"
        assert exporter.grafana_url == "http://localhost:3000"
        
    @patch('aiohttp.ClientSession.post')
    async def test_create_dashboard(self, mock_post):
        """Test de création de dashboard."""
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(
            return_value={"id": 123, "uid": "test_uid"}
        )
        
        exporter = GrafanaMultiTenantExporter(
            tenant_id="test_tenant",
            grafana_url="http://localhost:3000",
            api_key="test_key"
        )
        
        dashboard_id = await exporter.create_ai_metrics_dashboard()
        assert dashboard_id == 123


@pytest.mark.asyncio
class TestElasticsearchExporter:
    """Tests pour l'exportateur Elasticsearch."""
    
    async def test_exporter_creation(self):
        """Test de création de l'exportateur."""
        exporter = ElasticsearchMetricsExporter(
            tenant_id="test_tenant",
            hosts=["http://localhost:9200"]
        )
        
        assert exporter.tenant_id == "test_tenant"
        assert len(exporter.hosts) == 1


@pytest.mark.asyncio
class TestInfluxDBExporter:
    """Tests pour l'exportateur InfluxDB."""
    
    async def test_exporter_creation(self):
        """Test de création de l'exportateur."""
        exporter = InfluxDBMetricsExporter(
            tenant_id="test_tenant",
            url="http://localhost:8086",
            token="test_token",
            org="test_org"
        )
        
        assert exporter.tenant_id == "test_tenant"
        assert exporter.url == "http://localhost:8086"


@pytest.mark.asyncio
class TestCustomExporters:
    """Tests pour les exportateurs tiers."""
    
    async def test_datadog_exporter_creation(self):
        """Test de création de l'exportateur Datadog."""
        exporter = DatadogExporter(
            tenant_id="test_tenant",
            api_key="test_key",
            app_key="test_app_key"
        )
        
        assert exporter.tenant_id == "test_tenant"
        assert exporter.api_key == "test_key"
        
    async def test_newrelic_exporter_creation(self):
        """Test de création de l'exportateur New Relic."""
        exporter = NewRelicExporter(
            tenant_id="test_tenant",
            api_key="test_key",
            account_id="123456"
        )
        
        assert exporter.tenant_id == "test_tenant"
        assert exporter.api_key == "test_key"
        
    async def test_splunk_exporter_creation(self):
        """Test de création de l'exportateur Splunk."""
        exporter = SplunkExporter(
            tenant_id="test_tenant",
            hec_url="http://localhost:8088",
            hec_token="test_token"
        )
        
        assert exporter.tenant_id == "test_tenant"
        assert exporter.hec_url == "http://localhost:8088"


@pytest.mark.asyncio
class TestBatchExporter:
    """Tests pour l'exportateur batch."""
    
    async def test_exporter_creation(self):
        """Test de création de l'exportateur batch."""
        mock_exporter = Mock()
        
        exporter = BatchMetricsExporter(
            tenant_id="test_tenant",
            target_exporters=[mock_exporter],
            batch_size=10
        )
        
        assert exporter.tenant_id == "test_tenant"
        assert exporter.batch_size == 10
        
    async def test_batch_processing(self):
        """Test de traitement par lots."""
        mock_exporter = AsyncMock()
        
        exporter = BatchMetricsExporter(
            tenant_id="test_tenant",
            target_exporters=[mock_exporter],
            batch_size=2
        )
        
        metrics = [
            {"metric": "test1", "value": 1},
            {"metric": "test2", "value": 2},
            {"metric": "test3", "value": 3}
        ]
        
        await exporter.export_metrics(metrics)
        
        # Vérifier que l'exportateur cible a été appelé
        assert mock_exporter.export_metrics.called


@pytest.mark.asyncio
class TestStreamingExporter:
    """Tests pour l'exportateur streaming."""
    
    async def test_exporter_creation(self):
        """Test de création de l'exportateur streaming."""
        exporter = StreamingMetricsExporter(
            tenant_id="test_tenant",
            websocket_url="ws://localhost:8080/ws",
            sse_url="http://localhost:8080/sse"
        )
        
        assert exporter.tenant_id == "test_tenant"
        assert exporter.websocket_url == "ws://localhost:8080/ws"


@pytest.mark.asyncio
class TestMetricsExportManager:
    """Tests pour le gestionnaire d'export."""
    
    async def test_manager_creation(self):
        """Test de création du gestionnaire."""
        config = ExporterConfiguration(tenant_id="test_tenant")
        manager = MetricsExportManager(config)
        
        assert manager.config.tenant_id == "test_tenant"
        
    async def test_export_metrics_integration(self):
        """Test d'intégration d'export de métriques."""
        config = ExporterConfiguration(tenant_id="test_tenant")
        
        # Désactiver tous les exportateurs pour éviter les connexions réelles
        config.prometheus["enabled"] = False
        config.grafana["enabled"] = False
        config.elasticsearch["enabled"] = False
        config.influxdb["enabled"] = False
        config.batch["enabled"] = False
        config.streaming["enabled"] = False
        
        manager = MetricsExportManager(config)
        
        metrics = SpotifyAIMetrics(
            tenant_id="test_tenant",
            timestamp=datetime.now(timezone.utc)
        )
        
        results = await manager.export_metrics(metrics)
        
        # Aucun exportateur activé, donc résultats vides
        assert len(results) == 0


class TestPerformance:
    """Tests de performance."""
    
    def test_metrics_generation_performance(self):
        """Test de performance de génération de métriques."""
        generator = SpotifyAIMetricsGenerator("test_tenant")
        
        start_time = time.time()
        
        # Générer 1000 métriques
        for _ in range(1000):
            generator.generate_realistic_metrics()
            
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Vérifier que la génération est rapide (< 1 seconde pour 1000 métriques)
        assert generation_time < 1.0
        
        metrics_per_second = 1000 / generation_time
        logger.info(f"Metrics generation performance: {metrics_per_second:.2f} metrics/second")
        
    def test_prometheus_format_performance(self):
        """Test de performance de formatage Prometheus."""
        metrics = SpotifyAIMetrics(
            tenant_id="test_tenant",
            timestamp=datetime.now(timezone.utc),
            ai_inference_requests=100,
            ai_model_accuracy=0.95
        )
        
        start_time = time.time()
        
        # Formater 1000 fois
        for _ in range(1000):
            metrics.to_prometheus_format()
            
        end_time = time.time()
        formatting_time = end_time - start_time
        
        # Vérifier que le formatage est rapide
        assert formatting_time < 1.0
        
        formats_per_second = 1000 / formatting_time
        logger.info(f"Prometheus formatting performance: {formats_per_second:.2f} formats/second")


class TestErrorHandling:
    """Tests de gestion d'erreurs."""
    
    def test_invalid_tenant_id(self):
        """Test avec ID tenant invalide."""
        with pytest.raises(ValueError):
            SpotifyAIMetrics(
                tenant_id="",  # ID vide
                timestamp=datetime.now(timezone.utc)
            )
            
    def test_configuration_validation_errors(self):
        """Test des erreurs de validation de configuration."""
        config_manager = ConfigurationManager()
        
        # Configuration avec tenant_id trop court
        config = ExporterConfiguration(tenant_id="ab")  # Moins de 3 caractères
        errors = config_manager.validate_configuration(config)
        
        assert len(errors) > 0
        assert any("tenant_id" in error for error in errors)
        
    @pytest.mark.asyncio
    async def test_exporter_network_error_handling(self):
        """Test de gestion d'erreurs réseau."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simuler une erreur réseau
            mock_post.side_effect = Exception("Network error")
            
            exporter = PrometheusMultiTenantExporter(
                tenant_id="test_tenant",
                prometheus_url="http://localhost:9090"
            )
            
            metrics = [{"name": "test", "value": 1}]
            
            # L'export ne devrait pas lever d'exception
            try:
                await exporter.export_metrics(metrics)
            except Exception as e:
                pytest.fail(f"Export should handle network errors gracefully: {e}")


# Fixtures pour les tests
@pytest.fixture
def sample_metrics():
    """Fixture pour des métriques d'exemple."""
    return SpotifyAIMetrics(
        tenant_id="test_tenant",
        timestamp=datetime.now(timezone.utc),
        ai_inference_requests=100,
        ai_model_accuracy=0.95,
        recommendation_requests=50,
        tracks_analyzed=200
    )


@pytest.fixture
def sample_config():
    """Fixture pour une configuration d'exemple."""
    return ExporterConfiguration(
        tenant_id="test_tenant",
        environment=Environment.DEVELOPMENT
    )


@pytest.fixture
async def mock_prometheus_exporter():
    """Fixture pour un exportateur Prometheus mocké."""
    with patch('aiohttp.ClientSession') as mock_session:
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value.status = 200
        
        exporter = PrometheusMultiTenantExporter(
            tenant_id="test_tenant",
            prometheus_url="http://localhost:9090"
        )
        
        yield exporter


# Tests d'intégration
@pytest.mark.integration
@pytest.mark.asyncio
class TestIntegration:
    """Tests d'intégration nécessitant des services externes."""
    
    async def test_full_pipeline_integration(self):
        """Test d'intégration complète du pipeline."""
        # Ce test nécessiterait des services réels (Prometheus, Grafana, etc.)
        # À implémenter avec des containers Docker pour les tests CI/CD
        pytest.skip("Requires external services")
        
    async def test_multi_tenant_isolation(self):
        """Test d'isolation multi-tenant."""
        # Vérifier que les métriques de différents tenants sont bien isolées
        pytest.skip("Requires external services")


# Utilitaires pour les tests
class TestUtils:
    """Utilitaires pour les tests."""
    
    @staticmethod
    def create_test_metrics(tenant_id: str, count: int = 1) -> List[SpotifyAIMetrics]:
        """Crée des métriques de test."""
        generator = SpotifyAIMetricsGenerator(tenant_id)
        return [generator.generate_realistic_metrics() for _ in range(count)]
        
    @staticmethod
    def assert_metrics_valid(metrics: SpotifyAIMetrics):
        """Valide qu'une métrique est correcte."""
        assert metrics.tenant_id
        assert metrics.timestamp
        assert metrics.ai_model_accuracy >= 0 and metrics.ai_model_accuracy <= 1
        assert metrics.cache_hit_rate >= 0 and metrics.cache_hit_rate <= 1
        assert metrics.error_rate >= 0


if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--capture=no"
    ])
