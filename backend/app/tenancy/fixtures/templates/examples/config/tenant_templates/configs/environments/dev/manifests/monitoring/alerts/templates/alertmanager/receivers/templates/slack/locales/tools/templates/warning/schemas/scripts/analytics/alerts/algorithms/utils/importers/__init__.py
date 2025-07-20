"""
Spotify AI Agent - Ultra-Advanced Data Importers Module
======================================================

Industrialized, enterprise-grade data importers for multi-source data ingestion,
real-time streaming, batch processing, and comprehensive data transformation.

This module provides ultra-advanced importing capabilities for:
- Multi-source data ingestion (APIs, databases, files, streams)
- Real-time streaming data importers with Apache Kafka/Pulsar
- Batch processing with Apache Spark and distributed computing
- Music metadata extraction and audio analysis pipelines
- Social media and external platform integrations
- Data validation, cleansing, and transformation pipelines
- Schema evolution and data versioning systems
- Multi-tenant data isolation and security
- ETL/ELT orchestration with Apache Airflow integration
- Machine Learning feature store ingestion
- Analytics events and behavioral data collection
- Compliance and audit trail management

Author: Developed by Expert Team under Fahed Mlaiel's Leadership
Team: Lead Dev + AI Architect, Senior Backend Developer, ML Engineer,
      Data Engineer, Security Specialist, Microservices Architect
Version: 2.1.0
License: MIT
"""

from typing import Dict, Any, Optional, List, Union, AsyncGenerator
import logging
import asyncio
from datetime import datetime, timezone

# Core importer classes
from .audio_importer import (
    BaseAudioImporter,
    SpotifyAudioImporter,
    LastFMImporter,
    SoundCloudImporter,
    AudioFeatureExtractor,
    create_audio_importer
)

from .streaming_importer import (
    BaseStreamingImporter,
    KafkaStreamImporter,
    PulsarStreamImporter,
    RedisStreamImporter,
    WebSocketStreamImporter,
    EventHubImporter,
    create_streaming_importer
)

from .database_importer import (
    BaseDatabaseImporter,
    PostgreSQLImporter,
    MongoDBImporter,
    RedisImporter,
    ElasticsearchImporter,
    ClickHouseImporter,
    create_database_importer
)

from .api_importer import (
    BaseAPIImporter,
    SpotifyAPIImporter,
    SocialMediaImporter,
    RESTAPIImporter,
    GraphQLImporter,
    WebhookImporter,
    create_api_importer
)

from .file_importer import (
    BaseFileImporter,
    CSVImporter,
    JSONImporter,
    ParquetImporter,
    AvroImporter,
    S3FileImporter,
    create_file_importer
)

from .ml_feature_importer import (
    BaseMLFeatureImporter,
    FeatureStoreImporter,
    MLFlowImporter,
    TensorFlowDataImporter,
    HuggingFaceImporter,
    create_ml_feature_importer
)

from .analytics_importer import (
    BaseAnalyticsImporter,
    GoogleAnalyticsImporter,
    MixpanelImporter,
    SegmentImporter,
    AmplitudeImporter,
    create_analytics_importer
)

from .compliance_importer import (
    BaseComplianceImporter,
    GDPRDataImporter,
    AuditLogImporter,
    ComplianceReportImporter,
    create_compliance_importer
)

# Version information
__version__ = "2.1.0"
__author__ = "Expert Team under Fahed Mlaiel's Leadership"
__license__ = "MIT"

# Importer registry for factory pattern
IMPORTER_REGISTRY: Dict[str, Any] = {
    # Audio importers
    'audio': BaseAudioImporter,
    'spotify_audio': SpotifyAudioImporter,
    'lastfm': LastFMImporter,
    'soundcloud': SoundCloudImporter,
    'audio_features': AudioFeatureExtractor,
    
    # Streaming importers
    'streaming': BaseStreamingImporter,
    'kafka': KafkaStreamImporter,
    'pulsar': PulsarStreamImporter,
    'redis_stream': RedisStreamImporter,
    'websocket': WebSocketStreamImporter,
    'eventhub': EventHubImporter,
    
    # Database importers
    'database': BaseDatabaseImporter,
    'postgresql': PostgreSQLImporter,
    'mongodb': MongoDBImporter,
    'redis': RedisImporter,
    'elasticsearch': ElasticsearchImporter,
    'clickhouse': ClickHouseImporter,
    
    # API importers
    'api': BaseAPIImporter,
    'spotify_api': SpotifyAPIImporter,
    'social_media': SocialMediaImporter,
    'rest_api': RESTAPIImporter,
    'graphql': GraphQLImporter,
    'webhook': WebhookImporter,
    
    # File importers
    'file': BaseFileImporter,
    'csv': CSVImporter,
    'json': JSONImporter,
    'parquet': ParquetImporter,
    'avro': AvroImporter,
    's3': S3FileImporter,
    
    # ML feature importers
    'ml_features': BaseMLFeatureImporter,
    'feature_store': FeatureStoreImporter,
    'mlflow': MLFlowImporter,
    'tensorflow_data': TensorFlowDataImporter,
    'huggingface': HuggingFaceImporter,
    
    # Analytics importers
    'analytics': BaseAnalyticsImporter,
    'google_analytics': GoogleAnalyticsImporter,
    'mixpanel': MixpanelImporter,
    'segment': SegmentImporter,
    'amplitude': AmplitudeImporter,
    
    # Compliance importers
    'compliance': BaseComplianceImporter,
    'gdpr': GDPRDataImporter,
    'audit_log': AuditLogImporter,
    'compliance_report': ComplianceReportImporter,
}

# Factory functions registry
FACTORY_FUNCTIONS: Dict[str, Any] = {
    'audio': create_audio_importer,
    'streaming': create_streaming_importer,
    'database': create_database_importer,
    'api': create_api_importer,
    'file': create_file_importer,
    'ml_features': create_ml_feature_importer,
    'analytics': create_analytics_importer,
    'compliance': create_compliance_importer,
}


def get_importer(importer_type: str, 
                tenant_id: str, 
                config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Factory function to get appropriate importer instance.
    
    Args:
        importer_type: Type of importer to create
        tenant_id: Tenant identifier
        config: Optional configuration dictionary
        
    Returns:
        Configured importer instance
        
    Raises:
        ValueError: If importer type is not supported
    """
    if importer_type not in IMPORTER_REGISTRY:
        available_types = ", ".join(sorted(IMPORTER_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported importer type: {importer_type}. "
            f"Available types: {available_types}"
        )
    
    importer_class = IMPORTER_REGISTRY[importer_type]
    return importer_class(tenant_id, config or {})


def get_factory_function(category: str) -> Optional[Any]:
    """
    Get factory function for importer category.
    
    Args:
        category: Importer category (audio, streaming, database, etc.)
        
    Returns:
        Factory function or None if not found
    """
    return FACTORY_FUNCTIONS.get(category)


def list_available_importers() -> Dict[str, List[str]]:
    """
    List all available importers organized by category.
    
    Returns:
        Dictionary with categories and their available importers
    """
    categories = {
        'audio': ['audio', 'spotify_audio', 'lastfm', 'soundcloud', 'audio_features'],
        'streaming': ['streaming', 'kafka', 'pulsar', 'redis_stream', 'websocket', 'eventhub'],
        'database': ['database', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'clickhouse'],
        'api': ['api', 'spotify_api', 'social_media', 'rest_api', 'graphql', 'webhook'],
        'file': ['file', 'csv', 'json', 'parquet', 'avro', 's3'],
        'ml_features': ['ml_features', 'feature_store', 'mlflow', 'tensorflow_data', 'huggingface'],
        'analytics': ['analytics', 'google_analytics', 'mixpanel', 'segment', 'amplitude'],
        'compliance': ['compliance', 'gdpr', 'audit_log', 'compliance_report']
    }
    return categories


def get_importer_metadata() -> Dict[str, Any]:
    """
    Get metadata about the importers module.
    
    Returns:
        Dictionary containing module metadata
    """
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "total_importers": len(IMPORTER_REGISTRY),
        "categories": len(FACTORY_FUNCTIONS),
        "supported_sources": [
            "Spotify API", "Last.fm API", "SoundCloud API", "Social Media APIs",
            "Apache Kafka", "Apache Pulsar", "Redis Streams", "WebSocket",
            "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "ClickHouse",
            "CSV files", "JSON files", "Parquet files", "Avro files", "S3 buckets",
            "ML Feature Stores", "MLflow", "TensorFlow Datasets", "Hugging Face",
            "Google Analytics", "Mixpanel", "Segment", "Amplitude",
            "GDPR compliance data", "Audit logs", "Compliance reports"
        ],
        "features": [
            "Multi-source data ingestion", "Real-time streaming", "Batch processing",
            "Data validation and cleansing", "Schema evolution", "Multi-tenant isolation",
            "ETL/ELT orchestration", "Machine learning integration", "Analytics events",
            "Compliance and audit trails", "Performance optimization", "Error handling"
        ]
    }


async def orchestrate_import_pipeline(
    importers: List[Any],
    parallel: bool = True,
    max_concurrency: int = 10
) -> Dict[str, Any]:
    """
    Orchestrate multiple importers in a pipeline.
    
    Args:
        importers: List of configured importer instances
        parallel: Whether to run importers in parallel
        max_concurrency: Maximum number of concurrent importers
        
    Returns:
        Dictionary with import results and statistics
    """
    logger = logging.getLogger(__name__)
    start_time = datetime.now(timezone.utc)
    results = {}
    
    try:
        if parallel:
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def run_importer_with_semaphore(importer, idx):
                async with semaphore:
                    logger.info(f"Starting importer {idx}: {importer.__class__.__name__}")
                    try:
                        result = await importer.import_data()
                        return idx, {"status": "success", "result": result}
                    except Exception as e:
                        logger.error(f"Importer {idx} failed: {str(e)}")
                        return idx, {"status": "error", "error": str(e)}
            
            tasks = [
                run_importer_with_semaphore(importer, idx)
                for idx, importer in enumerate(importers)
            ]
            
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for idx, result in completed_results:
                if isinstance(result, Exception):
                    results[f"importer_{idx}"] = {"status": "error", "error": str(result)}
                else:
                    results[f"importer_{idx}"] = result[1]
        else:
            # Sequential execution
            for idx, importer in enumerate(importers):
                logger.info(f"Starting importer {idx}: {importer.__class__.__name__}")
                try:
                    result = await importer.import_data()
                    results[f"importer_{idx}"] = {"status": "success", "result": result}
                except Exception as e:
                    logger.error(f"Importer {idx} failed: {str(e)}")
                    results[f"importer_{idx}"] = {"status": "error", "error": str(e)}
        
        end_time = datetime.now(timezone.utc)
        execution_time = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        successful_imports = sum(1 for r in results.values() if r["status"] == "success")
        failed_imports = sum(1 for r in results.values() if r["status"] == "error")
        
        pipeline_stats = {
            "total_importers": len(importers),
            "successful_imports": successful_imports,
            "failed_imports": failed_imports,
            "success_rate": (successful_imports / len(importers)) * 100 if importers else 0,
            "execution_time_seconds": execution_time,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "parallel_execution": parallel,
            "max_concurrency": max_concurrency if parallel else 1
        }
        
        return {
            "pipeline_stats": pipeline_stats,
            "importer_results": results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Pipeline orchestration failed: {str(e)}")
        return {
            "pipeline_stats": {
                "total_importers": len(importers),
                "execution_time_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "status": "failed",
                "error": str(e)
            },
            "importer_results": results,
            "status": "failed"
        }


class ImporterHealthCheck:
    """Health check utility for importers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def check_importer_health(self, importer: Any) -> Dict[str, Any]:
        """
        Check health status of an importer.
        
        Args:
            importer: Importer instance to check
            
        Returns:
            Dictionary with health status information
        """
        health_info = {
            "importer_type": importer.__class__.__name__,
            "status": "unknown",
            "checks": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Check if importer has health check method
            if hasattr(importer, 'health_check'):
                check_result = await importer.health_check()
                health_info["status"] = "healthy" if check_result.get("healthy", False) else "unhealthy"
                health_info["checks"] = check_result.get("checks", {})
            else:
                # Basic connectivity check
                if hasattr(importer, 'test_connection'):
                    connection_test = await importer.test_connection()
                    health_info["status"] = "healthy" if connection_test else "unhealthy"
                    health_info["checks"]["connectivity"] = connection_test
                else:
                    health_info["status"] = "unknown"
                    health_info["checks"]["note"] = "No health check method available"
            
        except Exception as e:
            health_info["status"] = "error"
            health_info["error"] = str(e)
            self.logger.error(f"Health check failed for {importer.__class__.__name__}: {str(e)}")
        
        return health_info
    
    async def check_all_importers_health(self, importers: List[Any]) -> Dict[str, Any]:
        """
        Check health status of multiple importers.
        
        Args:
            importers: List of importer instances
            
        Returns:
            Dictionary with overall health summary
        """
        health_results = []
        
        for importer in importers:
            health_info = await self.check_importer_health(importer)
            health_results.append(health_info)
        
        # Calculate overall health statistics
        total_importers = len(importers)
        healthy_count = sum(1 for r in health_results if r["status"] == "healthy")
        unhealthy_count = sum(1 for r in health_results if r["status"] == "unhealthy")
        error_count = sum(1 for r in health_results if r["status"] == "error")
        unknown_count = sum(1 for r in health_results if r["status"] == "unknown")
        
        overall_status = "healthy"
        if error_count > 0 or unhealthy_count > 0:
            overall_status = "degraded" if healthy_count > 0 else "unhealthy"
        elif unknown_count == total_importers:
            overall_status = "unknown"
        
        return {
            "overall_status": overall_status,
            "summary": {
                "total_importers": total_importers,
                "healthy": healthy_count,
                "unhealthy": unhealthy_count,
                "error": error_count,
                "unknown": unknown_count,
                "health_percentage": (healthy_count / total_importers) * 100 if total_importers > 0 else 0
            },
            "individual_results": health_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Module-level exports
__all__ = [
    # Core classes
    'BaseAudioImporter', 'SpotifyAudioImporter', 'LastFMImporter', 'SoundCloudImporter',
    'AudioFeatureExtractor', 'BaseStreamingImporter', 'KafkaStreamImporter', 'PulsarStreamImporter',
    'RedisStreamImporter', 'WebSocketStreamImporter', 'EventHubImporter', 'BaseDatabaseImporter',
    'PostgreSQLImporter', 'MongoDBImporter', 'RedisImporter', 'ElasticsearchImporter',
    'ClickHouseImporter', 'BaseAPIImporter', 'SpotifyAPIImporter', 'SocialMediaImporter',
    'RESTAPIImporter', 'GraphQLImporter', 'WebhookImporter', 'BaseFileImporter',
    'CSVImporter', 'JSONImporter', 'ParquetImporter', 'AvroImporter', 'S3FileImporter',
    'BaseMLFeatureImporter', 'FeatureStoreImporter', 'MLFlowImporter', 'TensorFlowDataImporter',
    'HuggingFaceImporter', 'BaseAnalyticsImporter', 'GoogleAnalyticsImporter', 'MixpanelImporter',
    'SegmentImporter', 'AmplitudeImporter', 'BaseComplianceImporter', 'GDPRDataImporter',
    'AuditLogImporter', 'ComplianceReportImporter',
    
    # Factory functions
    'create_audio_importer', 'create_streaming_importer', 'create_database_importer',
    'create_api_importer', 'create_file_importer', 'create_ml_feature_importer',
    'create_analytics_importer', 'create_compliance_importer',
    
    # Utility functions
    'get_importer', 'get_factory_function', 'list_available_importers',
    'get_importer_metadata', 'orchestrate_import_pipeline',
    
    # Utility classes
    'ImporterHealthCheck',
    
    # Registries
    'IMPORTER_REGISTRY', 'FACTORY_FUNCTIONS',
    
    # Metadata
    '__version__', '__author__', '__license__'
]

# Initialize logging for the module
logging.getLogger(__name__).info(
    f"Spotify AI Agent Importers Module v{__version__} initialized with "
    f"{len(IMPORTER_REGISTRY)} importers across {len(FACTORY_FUNCTIONS)} categories"
)
