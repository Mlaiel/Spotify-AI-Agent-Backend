#!/usr/bin/env python3
"""
Spotify AI Agent - Metrics Exporters Example Usage
==================================================

Exemple complet d'utilisation de tous les exportateurs de métriques
dans un scénario réel de production multi-tenant.

Author: Fahed Mlaiel & Spotify AI Team
Date: 2024
"""

import asyncio
import json
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Importation des exportateurs
from .prometheus_exporter import PrometheusMultiTenantExporter
from .grafana_exporter import GrafanaMultiTenantExporter
from .elastic_exporter import ElasticsearchMetricsExporter
from .influxdb_exporter import InfluxDBMetricsExporter
from .custom_exporter import DatadogExporter, NewRelicExporter, SplunkExporter
from .batch_exporter import BatchMetricsExporter
from .streaming_exporter import StreamingMetricsExporter
from .config import ConfigurationManager, ExporterConfiguration, Environment

import structlog

# Configuration du logging
logger = structlog.get_logger(__name__)


@dataclass
class SpotifyAIMetrics:
    """Modèle de métriques pour Spotify AI."""
    
    # Identifiants
    tenant_id: str
    timestamp: datetime
    service: str = "spotify-ai-agent"
    
    # Métriques d'inférence AI
    ai_inference_requests: int = 0
    ai_inference_duration_ms: float = 0.0
    ai_model_accuracy: float = 0.0
    ai_model_confidence: float = 0.0
    ai_gpu_utilization: float = 0.0
    ai_memory_usage_mb: float = 0.0
    
    # Métriques de recommandation
    recommendation_requests: int = 0
    recommendation_hit_rate: float = 0.0
    recommendation_diversity_score: float = 0.0
    personalization_score: float = 0.0
    
    # Métriques métier Spotify
    tracks_analyzed: int = 0
    playlists_generated: int = 0
    user_interactions: int = 0
    streaming_quality_score: float = 0.0
    audio_feature_extraction_time: float = 0.0
    
    # Métriques de performance
    api_response_time_ms: float = 0.0
    database_query_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    
    # Métriques de coût
    compute_cost_usd: float = 0.0
    storage_cost_usd: float = 0.0
    api_calls_cost_usd: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour export."""
        return {
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp.isoformat(),
            "service": self.service,
            "ai_inference_requests": self.ai_inference_requests,
            "ai_inference_duration_ms": self.ai_inference_duration_ms,
            "ai_model_accuracy": self.ai_model_accuracy,
            "ai_model_confidence": self.ai_model_confidence,
            "ai_gpu_utilization": self.ai_gpu_utilization,
            "ai_memory_usage_mb": self.ai_memory_usage_mb,
            "recommendation_requests": self.recommendation_requests,
            "recommendation_hit_rate": self.recommendation_hit_rate,
            "recommendation_diversity_score": self.recommendation_diversity_score,
            "personalization_score": self.personalization_score,
            "tracks_analyzed": self.tracks_analyzed,
            "playlists_generated": self.playlists_generated,
            "user_interactions": self.user_interactions,
            "streaming_quality_score": self.streaming_quality_score,
            "audio_feature_extraction_time": self.audio_feature_extraction_time,
            "api_response_time_ms": self.api_response_time_ms,
            "database_query_time_ms": self.database_query_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "error_rate": self.error_rate,
            "throughput_rps": self.throughput_rps,
            "compute_cost_usd": self.compute_cost_usd,
            "storage_cost_usd": self.storage_cost_usd,
            "api_calls_cost_usd": self.api_calls_cost_usd
        }
        
    def to_prometheus_format(self) -> List[Dict[str, Any]]:
        """Formate pour Prometheus."""
        base_labels = {
            "tenant_id": self.tenant_id,
            "service": self.service
        }
        
        metrics = []
        
        # Métriques counter
        counters = [
            ("ai_inference_requests_total", self.ai_inference_requests),
            ("recommendation_requests_total", self.recommendation_requests),
            ("tracks_analyzed_total", self.tracks_analyzed),
            ("playlists_generated_total", self.playlists_generated),
            ("user_interactions_total", self.user_interactions)
        ]
        
        for name, value in counters:
            metrics.append({
                "name": name,
                "type": "counter",
                "value": value,
                "labels": base_labels,
                "timestamp": self.timestamp
            })
            
        # Métriques gauge
        gauges = [
            ("ai_inference_duration_ms", self.ai_inference_duration_ms),
            ("ai_model_accuracy", self.ai_model_accuracy),
            ("ai_model_confidence", self.ai_model_confidence),
            ("ai_gpu_utilization_percent", self.ai_gpu_utilization),
            ("ai_memory_usage_mb", self.ai_memory_usage_mb),
            ("recommendation_hit_rate", self.recommendation_hit_rate),
            ("recommendation_diversity_score", self.recommendation_diversity_score),
            ("personalization_score", self.personalization_score),
            ("streaming_quality_score", self.streaming_quality_score),
            ("audio_feature_extraction_time_ms", self.audio_feature_extraction_time),
            ("api_response_time_ms", self.api_response_time_ms),
            ("database_query_time_ms", self.database_query_time_ms),
            ("cache_hit_rate", self.cache_hit_rate),
            ("error_rate", self.error_rate),
            ("throughput_rps", self.throughput_rps),
            ("compute_cost_usd", self.compute_cost_usd),
            ("storage_cost_usd", self.storage_cost_usd),
            ("api_calls_cost_usd", self.api_calls_cost_usd)
        ]
        
        for name, value in gauges:
            metrics.append({
                "name": name,
                "type": "gauge",
                "value": value,
                "labels": base_labels,
                "timestamp": self.timestamp
            })
            
        return metrics


class SpotifyAIMetricsGenerator:
    """Générateur de métriques réalistes pour Spotify AI."""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.baseline_values = self._initialize_baseline()
        
    def _initialize_baseline(self) -> Dict[str, float]:
        """Initialise les valeurs de base."""
        return {
            "ai_inference_requests": 1000,
            "ai_inference_duration_ms": 150.0,
            "ai_model_accuracy": 0.87,
            "ai_model_confidence": 0.82,
            "ai_gpu_utilization": 65.0,
            "ai_memory_usage_mb": 2048.0,
            "recommendation_requests": 500,
            "recommendation_hit_rate": 0.75,
            "recommendation_diversity_score": 0.68,
            "personalization_score": 0.73,
            "tracks_analyzed": 800,
            "playlists_generated": 120,
            "user_interactions": 2500,
            "streaming_quality_score": 0.91,
            "audio_feature_extraction_time": 45.0,
            "api_response_time_ms": 85.0,
            "database_query_time_ms": 25.0,
            "cache_hit_rate": 0.85,
            "error_rate": 0.02,
            "throughput_rps": 150.0,
            "compute_cost_usd": 12.50,
            "storage_cost_usd": 3.20,
            "api_calls_cost_usd": 0.75
        }
    
    def generate_realistic_metrics(self) -> SpotifyAIMetrics:
        """Génère des métriques réalistes avec variations."""
        now = datetime.now(timezone.utc)
        
        # Simulation d'variations temporelles
        hour = now.hour
        is_peak_hour = 8 <= hour <= 22  # Heures de pointe
        peak_multiplier = 1.5 if is_peak_hour else 0.7
        
        # Simulation de week-end
        is_weekend = now.weekday() >= 5
        weekend_multiplier = 1.2 if is_weekend else 1.0
        
        # Variation aléatoire
        variance = 0.15
        
        def vary_value(base_value: float, multiplier: float = 1.0) -> float:
            """Applique des variations à une valeur."""
            variation = random.uniform(1 - variance, 1 + variance)
            return base_value * peak_multiplier * weekend_multiplier * multiplier * variation
        
        return SpotifyAIMetrics(
            tenant_id=self.tenant_id,
            timestamp=now,
            ai_inference_requests=int(vary_value(self.baseline_values["ai_inference_requests"])),
            ai_inference_duration_ms=vary_value(self.baseline_values["ai_inference_duration_ms"]),
            ai_model_accuracy=min(1.0, vary_value(self.baseline_values["ai_model_accuracy"], 0.98)),
            ai_model_confidence=min(1.0, vary_value(self.baseline_values["ai_model_confidence"], 0.98)),
            ai_gpu_utilization=min(100.0, vary_value(self.baseline_values["ai_gpu_utilization"])),
            ai_memory_usage_mb=vary_value(self.baseline_values["ai_memory_usage_mb"]),
            recommendation_requests=int(vary_value(self.baseline_values["recommendation_requests"])),
            recommendation_hit_rate=min(1.0, vary_value(self.baseline_values["recommendation_hit_rate"], 0.98)),
            recommendation_diversity_score=min(1.0, vary_value(self.baseline_values["recommendation_diversity_score"], 0.98)),
            personalization_score=min(1.0, vary_value(self.baseline_values["personalization_score"], 0.98)),
            tracks_analyzed=int(vary_value(self.baseline_values["tracks_analyzed"])),
            playlists_generated=int(vary_value(self.baseline_values["playlists_generated"])),
            user_interactions=int(vary_value(self.baseline_values["user_interactions"])),
            streaming_quality_score=min(1.0, vary_value(self.baseline_values["streaming_quality_score"], 0.99)),
            audio_feature_extraction_time=vary_value(self.baseline_values["audio_feature_extraction_time"]),
            api_response_time_ms=vary_value(self.baseline_values["api_response_time_ms"]),
            database_query_time_ms=vary_value(self.baseline_values["database_query_time_ms"]),
            cache_hit_rate=min(1.0, vary_value(self.baseline_values["cache_hit_rate"], 0.98)),
            error_rate=max(0.0, vary_value(self.baseline_values["error_rate"], 1.2)),
            throughput_rps=vary_value(self.baseline_values["throughput_rps"]),
            compute_cost_usd=vary_value(self.baseline_values["compute_cost_usd"]),
            storage_cost_usd=vary_value(self.baseline_values["storage_cost_usd"]),
            api_calls_cost_usd=vary_value(self.baseline_values["api_calls_cost_usd"])
        )


class MetricsExportManager:
    """Gestionnaire principal pour tous les exportateurs."""
    
    def __init__(self, config: ExporterConfiguration):
        self.config = config
        self.exporters = {}
        self._initialize_exporters()
        
    def _initialize_exporters(self):
        """Initialise tous les exportateurs selon la configuration."""
        try:
            # Prometheus
            if self.config.prometheus.get("enabled", False):
                self.exporters["prometheus"] = PrometheusMultiTenantExporter(
                    tenant_id=self.config.tenant_id,
                    prometheus_url=self.config.prometheus["url"],
                    encryption_enabled=self.config.prometheus.get("encryption_enabled", False),
                    compression_enabled=self.config.prometheus.get("compression_enabled", False),
                    rate_limit_per_second=self.config.prometheus.get("rate_limit_per_second", 1000)
                )
                
            # Grafana
            if self.config.grafana.get("enabled", False):
                self.exporters["grafana"] = GrafanaMultiTenantExporter(
                    tenant_id=self.config.tenant_id,
                    grafana_url=self.config.grafana["url"],
                    api_key=self.config.grafana.get("api_key", ""),
                    org_id=self.config.grafana.get("org_id", 1)
                )
                
            # Elasticsearch
            if self.config.elasticsearch.get("enabled", False):
                self.exporters["elasticsearch"] = ElasticsearchMetricsExporter(
                    tenant_id=self.config.tenant_id,
                    hosts=self.config.elasticsearch["hosts"],
                    username=self.config.elasticsearch.get("username", ""),
                    password=self.config.elasticsearch.get("password", ""),
                    index_prefix=self.config.elasticsearch.get("index_prefix", "spotify-ai-metrics")
                )
                
            # InfluxDB
            if self.config.influxdb.get("enabled", False):
                self.exporters["influxdb"] = InfluxDBMetricsExporter(
                    tenant_id=self.config.tenant_id,
                    url=self.config.influxdb["url"],
                    token=self.config.influxdb.get("token", ""),
                    org=self.config.influxdb.get("org", "spotify-ai"),
                    bucket_prefix=self.config.influxdb.get("bucket_prefix", "spotify-ai")
                )
                
            # Third-party exporters
            if self.config.third_party.get("datadog", {}).get("enabled", False):
                datadog_config = self.config.third_party["datadog"]
                self.exporters["datadog"] = DatadogExporter(
                    tenant_id=self.config.tenant_id,
                    api_key=datadog_config.get("api_key", ""),
                    app_key=datadog_config.get("app_key", ""),
                    site=datadog_config.get("site", "datadoghq.com")
                )
                
            if self.config.third_party.get("newrelic", {}).get("enabled", False):
                newrelic_config = self.config.third_party["newrelic"]
                self.exporters["newrelic"] = NewRelicExporter(
                    tenant_id=self.config.tenant_id,
                    api_key=newrelic_config.get("api_key", ""),
                    account_id=newrelic_config.get("account_id", "")
                )
                
            if self.config.third_party.get("splunk", {}).get("enabled", False):
                splunk_config = self.config.third_party["splunk"]
                self.exporters["splunk"] = SplunkExporter(
                    tenant_id=self.config.tenant_id,
                    hec_url=splunk_config.get("hec_url", ""),
                    hec_token=splunk_config.get("hec_token", ""),
                    index=splunk_config.get("index", "spotify_ai_metrics")
                )
                
            # Batch exporter
            if self.config.batch.get("enabled", False):
                self.exporters["batch"] = BatchMetricsExporter(
                    tenant_id=self.config.tenant_id,
                    target_exporters=list(self.exporters.values()),
                    batch_size=self.config.batch.get("batch_size", 1000),
                    batch_timeout=self.config.batch.get("batch_timeout", 30.0)
                )
                
            # Streaming exporter
            if self.config.streaming.get("enabled", False):
                self.exporters["streaming"] = StreamingMetricsExporter(
                    tenant_id=self.config.tenant_id,
                    websocket_url=self.config.streaming.get("protocols", {}).get("websocket", {}).get("url", ""),
                    sse_url=self.config.streaming.get("protocols", {}).get("sse", {}).get("url", ""),
                    mqtt_url=self.config.streaming.get("protocols", {}).get("mqtt", {}).get("url", "")
                )
                
            logger.info(f"Initialized {len(self.exporters)} exporters", exporters=list(self.exporters.keys()))
            
        except Exception as e:
            logger.error(f"Failed to initialize exporters: {e}")
            
    async def export_metrics(self, metrics: SpotifyAIMetrics) -> Dict[str, bool]:
        """Exporte les métriques vers tous les exportateurs activés."""
        results = {}
        
        for name, exporter in self.exporters.items():
            try:
                if hasattr(exporter, 'export_metrics'):
                    if name == "prometheus":
                        prometheus_metrics = metrics.to_prometheus_format()
                        await exporter.export_metrics(prometheus_metrics)
                    else:
                        await exporter.export_metrics([metrics.to_dict()])
                        
                    results[name] = True
                    logger.debug(f"Successfully exported to {name}")
                    
                else:
                    logger.warning(f"Exporter {name} does not support export_metrics")
                    results[name] = False
                    
            except Exception as e:
                logger.error(f"Failed to export to {name}: {e}")
                results[name] = False
                
        return results
        
    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """Vérifie l'état de santé de tous les exportateurs."""
        health_status = {}
        
        for name, exporter in self.exporters.items():
            try:
                if hasattr(exporter, 'health_check'):
                    status = await exporter.health_check()
                    health_status[name] = status
                else:
                    health_status[name] = {
                        "status": "unknown",
                        "message": "Health check not implemented"
                    }
                    
            except Exception as e:
                health_status[name] = {
                    "status": "error",
                    "message": str(e)
                }
                
        return health_status
        
    async def close(self):
        """Ferme tous les exportateurs."""
        for name, exporter in self.exporters.items():
            try:
                if hasattr(exporter, 'close'):
                    await exporter.close()
                logger.debug(f"Closed exporter {name}")
            except Exception as e:
                logger.error(f"Failed to close exporter {name}: {e}")


async def example_single_tenant_usage():
    """Exemple d'utilisation pour un seul tenant."""
    logger.info("=== Single Tenant Example ===")
    
    tenant_id = "spotify_artist_daft_punk"
    
    # Charger la configuration
    config_manager = ConfigurationManager()
    config = config_manager.load_configuration(tenant_id)
    
    # Valider la configuration
    errors = config_manager.validate_configuration(config)
    if errors:
        logger.error("Configuration errors", errors=errors)
        return
        
    # Initialiser le gestionnaire d'export
    export_manager = MetricsExportManager(config)
    
    # Vérifier l'état de santé
    health = await export_manager.health_check()
    logger.info("Health check results", health=health)
    
    # Générer et exporter des métriques
    metrics_generator = SpotifyAIMetricsGenerator(tenant_id)
    
    for i in range(5):
        metrics = metrics_generator.generate_realistic_metrics()
        
        logger.info(f"Generated metrics #{i+1}", 
                   ai_requests=metrics.ai_inference_requests,
                   recommendations=metrics.recommendation_requests,
                   accuracy=metrics.ai_model_accuracy)
        
        results = await export_manager.export_metrics(metrics)
        logger.info("Export results", results=results)
        
        await asyncio.sleep(2)
        
    # Nettoyer
    await export_manager.close()
    logger.info("Single tenant example completed")


async def example_multi_tenant_usage():
    """Exemple d'utilisation multi-tenant."""
    logger.info("=== Multi-Tenant Example ===")
    
    tenants = [
        "spotify_artist_daft_punk",
        "spotify_label_columbia_records", 
        "spotify_playlist_discover_weekly",
        "spotify_podcast_joe_rogan"
    ]
    
    managers = {}
    generators = {}
    
    # Initialiser pour chaque tenant
    for tenant_id in tenants:
        config_manager = ConfigurationManager()
        config = config_manager.load_configuration(tenant_id)
        
        managers[tenant_id] = MetricsExportManager(config)
        generators[tenant_id] = SpotifyAIMetricsGenerator(tenant_id)
        
    logger.info(f"Initialized {len(managers)} tenant managers")
    
    # Simuler l'export en parallèle
    async def export_for_tenant(tenant_id: str, rounds: int = 3):
        """Exporte des métriques pour un tenant."""
        manager = managers[tenant_id]
        generator = generators[tenant_id]
        
        for i in range(rounds):
            metrics = generator.generate_realistic_metrics()
            results = await manager.export_metrics(metrics)
            
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"Tenant {tenant_id} round {i+1}", 
                       exports_successful=success_count, 
                       total_exporters=len(results))
            
            await asyncio.sleep(1)
            
    # Exécuter en parallèle pour tous les tenants
    tasks = [export_for_tenant(tenant_id) for tenant_id in tenants]
    await asyncio.gather(*tasks)
    
    # Nettoyer
    for manager in managers.values():
        await manager.close()
        
    logger.info("Multi-tenant example completed")


async def example_batch_processing():
    """Exemple de traitement par lots."""
    logger.info("=== Batch Processing Example ===")
    
    tenant_id = "spotify_batch_processing_demo"
    
    # Configuration optimisée pour le batch
    config_manager = ConfigurationManager()
    config = config_manager.load_configuration(tenant_id)
    
    # Adapter pour le batch
    config.batch["enabled"] = True
    config.batch["batch_size"] = 100
    config.batch["parallel_workers"] = 4
    
    export_manager = MetricsExportManager(config)
    metrics_generator = SpotifyAIMetricsGenerator(tenant_id)
    
    # Générer un lot de métriques
    batch_metrics = []
    for i in range(500):  # 500 métriques
        metrics = metrics_generator.generate_realistic_metrics()
        batch_metrics.append(metrics)
        
    logger.info(f"Generated batch of {len(batch_metrics)} metrics")
    
    # Exporter par lots
    batch_exporter = export_manager.exporters.get("batch")
    if batch_exporter:
        start_time = time.time()
        
        for metrics in batch_metrics:
            results = await export_manager.export_metrics(metrics)
            
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("Batch processing completed", 
                   total_metrics=len(batch_metrics),
                   processing_time_seconds=processing_time,
                   metrics_per_second=len(batch_metrics) / processing_time)
    else:
        logger.error("Batch exporter not available")
        
    await export_manager.close()


async def example_streaming_demo():
    """Exemple de streaming en temps réel."""
    logger.info("=== Streaming Example ===")
    
    tenant_id = "spotify_streaming_demo"
    
    # Configuration pour le streaming
    config_manager = ConfigurationManager()
    config = config_manager.load_configuration(tenant_id)
    
    config.streaming["enabled"] = True
    config.streaming["protocols"]["websocket"]["enabled"] = True
    config.streaming["protocols"]["sse"]["enabled"] = True
    
    export_manager = MetricsExportManager(config)
    metrics_generator = SpotifyAIMetricsGenerator(tenant_id)
    
    # Simuler un flux de métriques en temps réel
    streaming_exporter = export_manager.exporters.get("streaming")
    if streaming_exporter:
        logger.info("Starting real-time metrics streaming")
        
        # Démarrer le streaming
        await streaming_exporter.start_streaming()
        
        # Envoyer des métriques en continu
        for i in range(20):
            metrics = metrics_generator.generate_realistic_metrics()
            
            # Stream vers WebSocket
            await streaming_exporter.stream_metrics([metrics.to_dict()], "websocket")
            
            # Stream vers SSE
            await streaming_exporter.stream_metrics([metrics.to_dict()], "sse")
            
            logger.info(f"Streamed metrics #{i+1}", protocol="websocket+sse")
            
            await asyncio.sleep(0.5)  # 2 métriques par seconde
            
        # Arrêter le streaming
        await streaming_exporter.stop_streaming()
        logger.info("Streaming completed")
    else:
        logger.error("Streaming exporter not available")
        
    await export_manager.close()


async def example_performance_benchmark():
    """Benchmark de performance des exportateurs."""
    logger.info("=== Performance Benchmark ===")
    
    tenant_id = "spotify_benchmark"
    
    config_manager = ConfigurationManager()
    config = config_manager.load_configuration(tenant_id)
    
    export_manager = MetricsExportManager(config)
    metrics_generator = SpotifyAIMetricsGenerator(tenant_id)
    
    # Test de différentes tailles de lots
    batch_sizes = [1, 10, 50, 100, 500]
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        
        # Générer le lot
        metrics_batch = []
        for _ in range(batch_size):
            metrics_batch.append(metrics_generator.generate_realistic_metrics())
            
        # Mesurer le temps d'export
        start_time = time.time()
        
        for metrics in metrics_batch:
            await export_manager.export_metrics(metrics)
            
        end_time = time.time()
        
        # Calculer les performances
        total_time = end_time - start_time
        metrics_per_second = batch_size / total_time if total_time > 0 else 0
        
        logger.info("Benchmark result",
                   batch_size=batch_size,
                   total_time_seconds=total_time,
                   metrics_per_second=metrics_per_second)
                   
    await export_manager.close()


async def main():
    """Fonction principale - exécute tous les exemples."""
    logger.info("🎵 Spotify AI Agent - Metrics Exporters Demo 🎵")
    logger.info("=" * 60)
    
    try:
        # Exemple 1: Single tenant
        await example_single_tenant_usage()
        await asyncio.sleep(2)
        
        # Exemple 2: Multi-tenant
        await example_multi_tenant_usage()
        await asyncio.sleep(2)
        
        # Exemple 3: Batch processing
        await example_batch_processing()
        await asyncio.sleep(2)
        
        # Exemple 4: Streaming
        await example_streaming_demo()
        await asyncio.sleep(2)
        
        # Exemple 5: Performance benchmark
        await example_performance_benchmark()
        
        logger.info("🎉 All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Configuration du logging structuré
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Exécuter la démo
    asyncio.run(main())
