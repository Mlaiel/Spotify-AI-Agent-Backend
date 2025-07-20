"""
Advanced Custom Multi-Tenant Metrics Exporter Framework
======================================================

Framework extensible pour créer des exportateurs personnalisés avec
support multi-tenant et intégrations tierces.

Fonctionnalités:
- Base abstraite pour exportateurs
- Plugin architecture
- Intégrations tierces (Datadog, New Relic, etc.)
- Transformation de données
- Middleware personnalisé
- Monitoring intégré
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class ExporterType(Enum):
    """Types d'exportateurs supportés."""
    TIME_SERIES = "time_series"
    LOG_AGGREGATION = "log_aggregation"
    EVENT_STREAMING = "event_streaming"
    ANALYTICS = "analytics"
    ALERTING = "alerting"


@dataclass
class ExporterConfig:
    """Configuration de base pour un exportateur."""
    exporter_id: str
    exporter_type: ExporterType
    tenant_id: str
    enabled: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    rate_limit: int = 1000
    batch_size: int = 100
    buffer_size: int = 10000
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricPayload:
    """Payload de métrique unifié."""
    tenant_id: str
    metric_name: str
    metric_value: Union[int, float, str, bool]
    metric_type: str
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "spotify-ai-agent"


class BaseMetricsExporter(ABC):
    """
    Classe de base abstraite pour tous les exportateurs.
    
    Définit l'interface commune et les fonctionnalités de base
    pour tous les exportateurs de métriques.
    """
    
    def __init__(self, config: ExporterConfig):
        self.config = config
        self.is_initialized = False
        self.is_running = False
        
        # Buffer pour les métriques
        self.metric_buffer: List[MetricPayload] = []
        self.buffer_lock = asyncio.Lock()
        
        # Métriques internes
        self.stats = {
            'metrics_processed': 0,
            'exports_successful': 0,
            'exports_failed': 0,
            'last_export': None,
            'errors': []
        }
        
        # Middleware stack
        self.middleware: List[Callable] = []
        
    async def initialize(self):
        """Initialise l'exportateur."""
        try:
            await self._setup_exporter()
            self.is_initialized = True
            logger.info(
                f"Exporter {self.config.exporter_id} initialized",
                tenant_id=self.config.tenant_id,
                exporter_type=self.config.exporter_type.value
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize exporter {self.config.exporter_id}: {e}"
            )
            raise
            
    @abstractmethod
    async def _setup_exporter(self):
        """Configure l'exportateur spécifique."""
        pass
        
    @abstractmethod
    async def _export_metrics(self, metrics: List[MetricPayload]) -> bool:
        """Exporte les métriques vers la destination."""
        pass
        
    async def export_metric(self, metric: MetricPayload):
        """
        Exporte une métrique unique.
        
        Args:
            metric: Métrique à exporter
        """
        if not self.is_initialized:
            raise RuntimeError("Exporter not initialized")
            
        # Appliquer les middleware
        processed_metric = await self._apply_middleware(metric)
        
        async with self.buffer_lock:
            self.metric_buffer.append(processed_metric)
            
            # Vérifier si on doit vider le buffer
            if len(self.metric_buffer) >= self.config.batch_size:
                await self._flush_buffer()
                
    async def export_metrics(self, metrics: List[MetricPayload]):
        """
        Exporte plusieurs métriques.
        
        Args:
            metrics: Liste des métriques à exporter
        """
        for metric in metrics:
            await self.export_metric(metric)
            
    async def _flush_buffer(self):
        """Vide le buffer en exportant toutes les métriques."""
        if not self.metric_buffer:
            return
            
        try:
            # Copier et vider le buffer
            metrics_to_export = self.metric_buffer.copy()
            self.metric_buffer.clear()
            
            # Exporter par lots
            for i in range(0, len(metrics_to_export), self.config.batch_size):
                batch = metrics_to_export[i:i + self.config.batch_size]
                
                # Retry logic
                for attempt in range(self.config.retry_attempts):
                    try:
                        success = await self._export_metrics(batch)
                        if success:
                            self.stats['exports_successful'] += 1
                            self.stats['metrics_processed'] += len(batch)
                            break
                    except Exception as e:
                        if attempt == self.config.retry_attempts - 1:
                            self.stats['exports_failed'] += 1
                            self.stats['errors'].append({
                                'timestamp': datetime.now().isoformat(),
                                'error': str(e),
                                'batch_size': len(batch)
                            })
                            logger.error(
                                f"Failed to export metrics after {self.config.retry_attempts} attempts: {e}",
                                exporter_id=self.config.exporter_id
                            )
                        else:
                            await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                            
            self.stats['last_export'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(
                f"Failed to flush buffer for exporter {self.config.exporter_id}: {e}"
            )
            
    async def _apply_middleware(self, metric: MetricPayload) -> MetricPayload:
        """Applique les middleware à une métrique."""
        processed_metric = metric
        
        for middleware_func in self.middleware:
            try:
                processed_metric = await middleware_func(processed_metric)
            except Exception as e:
                logger.warning(
                    f"Middleware error in exporter {self.config.exporter_id}: {e}"
                )
                
        return processed_metric
        
    def add_middleware(self, middleware_func: Callable):
        """Ajoute un middleware au stack."""
        self.middleware.append(middleware_func)
        
    async def start(self):
        """Démarre l'exportateur."""
        if not self.is_initialized:
            await self.initialize()
            
        self.is_running = True
        logger.info(f"Exporter {self.config.exporter_id} started")
        
    async def stop(self):
        """Arrête l'exportateur."""
        self.is_running = False
        
        # Vider le buffer final
        async with self.buffer_lock:
            if self.metric_buffer:
                await self._flush_buffer()
                
        logger.info(f"Exporter {self.config.exporter_id} stopped")
        
    async def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'exportateur."""
        return {
            'config': {
                'exporter_id': self.config.exporter_id,
                'exporter_type': self.config.exporter_type.value,
                'tenant_id': self.config.tenant_id,
                'enabled': self.config.enabled
            },
            'status': {
                'initialized': self.is_initialized,
                'running': self.is_running,
                'buffer_size': len(self.metric_buffer)
            },
            'stats': self.stats
        }


class DatadogExporter(BaseMetricsExporter):
    """Exportateur pour Datadog."""
    
    def __init__(self, config: ExporterConfig, api_key: str, app_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.app_key = app_key
        self.base_url = "https://api.datadoghq.com/api/v1"
        
    async def _setup_exporter(self):
        """Configure l'exportateur Datadog."""
        # Vérifier la connexion API
        async with aiohttp.ClientSession() as session:
            headers = {
                'DD-API-KEY': self.api_key,
                'DD-APPLICATION-KEY': self.app_key,
                'Content-Type': 'application/json'
            }
            
            async with session.get(
                f"{self.base_url}/validate",
                headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Datadog API validation failed: {response.status}")
                    
    async def _export_metrics(self, metrics: List[MetricPayload]) -> bool:
        """Exporte vers Datadog."""
        try:
            datadog_metrics = []
            
            for metric in metrics:
                dd_metric = {
                    'metric': f"spotify.ai.{metric.metric_name}",
                    'points': [(int(metric.timestamp.timestamp()), metric.metric_value)],
                    'tags': [
                        f"tenant:{metric.tenant_id}",
                        f"source:{metric.source}"
                    ] + [f"{k}:{v}" for k, v in metric.labels.items()],
                    'type': self._convert_metric_type(metric.metric_type)
                }
                datadog_metrics.append(dd_metric)
                
            payload = {'series': datadog_metrics}
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'DD-API-KEY': self.api_key,
                    'DD-APPLICATION-KEY': self.app_key,
                    'Content-Type': 'application/json'
                }
                
                async with session.post(
                    f"{self.base_url}/series",
                    json=payload,
                    headers=headers
                ) as response:
                    return response.status == 202
                    
        except Exception as e:
            logger.error(f"Datadog export failed: {e}")
            return False
            
    def _convert_metric_type(self, metric_type: str) -> str:
        """Convertit le type de métrique pour Datadog."""
        type_mapping = {
            'gauge': 'gauge',
            'counter': 'count',
            'histogram': 'gauge'
        }
        return type_mapping.get(metric_type, 'gauge')


class NewRelicExporter(BaseMetricsExporter):
    """Exportateur pour New Relic."""
    
    def __init__(self, config: ExporterConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.base_url = "https://metric-api.newrelic.com/metric/v1"
        
    async def _setup_exporter(self):
        """Configure l'exportateur New Relic."""
        # Test de connectivité
        headers = {
            'Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        test_payload = [{
            'metrics': [{
                'name': 'spotify.ai.test',
                'type': 'gauge',
                'value': 1,
                'timestamp': int(datetime.now().timestamp() * 1000),
                'attributes': {
                    'tenant.id': self.config.tenant_id,
                    'test': 'connectivity'
                }
            }]
        }]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                json=test_payload,
                headers=headers
            ) as response:
                if response.status != 202:
                    raise Exception(f"New Relic API test failed: {response.status}")
                    
    async def _export_metrics(self, metrics: List[MetricPayload]) -> bool:
        """Exporte vers New Relic."""
        try:
            nr_metrics = []
            
            for metric in metrics:
                nr_metric = {
                    'name': f"spotify.ai.{metric.metric_name}",
                    'type': self._convert_metric_type(metric.metric_type),
                    'value': metric.metric_value,
                    'timestamp': int(metric.timestamp.timestamp() * 1000),
                    'attributes': {
                        'tenant.id': metric.tenant_id,
                        'source': metric.source,
                        **metric.labels
                    }
                }
                nr_metrics.append(nr_metric)
                
            payload = [{'metrics': nr_metrics}]
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Api-Key': self.api_key,
                    'Content-Type': 'application/json'
                }
                
                async with session.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                ) as response:
                    return response.status == 202
                    
        except Exception as e:
            logger.error(f"New Relic export failed: {e}")
            return False
            
    def _convert_metric_type(self, metric_type: str) -> str:
        """Convertit le type de métrique pour New Relic."""
        type_mapping = {
            'gauge': 'gauge',
            'counter': 'count',
            'histogram': 'gauge'
        }
        return type_mapping.get(metric_type, 'gauge')


class SplunkExporter(BaseMetricsExporter):
    """Exportateur pour Splunk."""
    
    def __init__(self, config: ExporterConfig, hec_url: str, hec_token: str):
        super().__init__(config)
        self.hec_url = hec_url.rstrip('/')
        self.hec_token = hec_token
        
    async def _setup_exporter(self):
        """Configure l'exportateur Splunk."""
        # Test HEC endpoint
        test_event = {
            'time': datetime.now().timestamp(),
            'event': {
                'metric_name': 'spotify.ai.test',
                'value': 1,
                'tenant_id': self.config.tenant_id
            },
            'sourcetype': 'spotify:ai:metrics'
        }
        
        headers = {
            'Authorization': f'Splunk {self.hec_token}',
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.hec_url}/services/collector",
                json=test_event,
                headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"Splunk HEC test failed: {response.status}")
                    
    async def _export_metrics(self, metrics: List[MetricPayload]) -> bool:
        """Exporte vers Splunk."""
        try:
            events = []
            
            for metric in metrics:
                event = {
                    'time': metric.timestamp.timestamp(),
                    'event': {
                        'metric_name': metric.metric_name,
                        'metric_value': metric.metric_value,
                        'metric_type': metric.metric_type,
                        'tenant_id': metric.tenant_id,
                        'source': metric.source,
                        'labels': metric.labels,
                        'metadata': metric.metadata
                    },
                    'sourcetype': 'spotify:ai:metrics',
                    'index': f'spotify_ai_{self.config.tenant_id}'
                }
                events.append(event)
                
            # Envoyer par lots
            headers = {
                'Authorization': f'Splunk {self.hec_token}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                # Splunk HEC peut prendre plusieurs événements séparés par des nouvelles lignes
                payload = '\n'.join(json.dumps(event) for event in events)
                
                async with session.post(
                    f"{self.hec_url}/services/collector",
                    data=payload,
                    headers=headers
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Splunk export failed: {e}")
            return False


class WebhookExporter(BaseMetricsExporter):
    """Exportateur générique webhook."""
    
    def __init__(
        self, 
        config: ExporterConfig, 
        webhook_url: str,
        headers: Optional[Dict[str, str]] = None,
        auth_token: Optional[str] = None
    ):
        super().__init__(config)
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.auth_token = auth_token
        
    async def _setup_exporter(self):
        """Configure l'exportateur webhook."""
        # Test webhook
        test_payload = {
            'test': True,
            'tenant_id': self.config.tenant_id,
            'timestamp': datetime.now().isoformat()
        }
        
        headers = self.headers.copy()
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        headers['Content-Type'] = 'application/json'
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.webhook_url,
                json=test_payload,
                headers=headers
            ) as response:
                if response.status >= 400:
                    raise Exception(f"Webhook test failed: {response.status}")
                    
    async def _export_metrics(self, metrics: List[MetricPayload]) -> bool:
        """Exporte vers webhook."""
        try:
            payload = {
                'tenant_id': self.config.tenant_id,
                'timestamp': datetime.now().isoformat(),
                'metrics': []
            }
            
            for metric in metrics:
                metric_data = {
                    'name': metric.metric_name,
                    'value': metric.metric_value,
                    'type': metric.metric_type,
                    'timestamp': metric.timestamp.isoformat(),
                    'labels': metric.labels,
                    'metadata': metric.metadata,
                    'source': metric.source
                }
                payload['metrics'].append(metric_data)
                
            headers = self.headers.copy()
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'
            headers['Content-Type'] = 'application/json'
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    return response.status < 400
                    
        except Exception as e:
            logger.error(f"Webhook export failed: {e}")
            return False


class CustomMetricsExporter:
    """
    Gestionnaire principal pour les exportateurs personnalisés.
    
    Gère multiple exportateurs et leur orchestration.
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.exporters: Dict[str, BaseMetricsExporter] = {}
        self.running = False
        
    def register_exporter(self, exporter: BaseMetricsExporter):
        """Enregistre un exportateur."""
        self.exporters[exporter.config.exporter_id] = exporter
        logger.info(
            f"Registered exporter {exporter.config.exporter_id}",
            tenant_id=self.tenant_id,
            exporter_type=exporter.config.exporter_type.value
        )
        
    def unregister_exporter(self, exporter_id: str):
        """Désenregistre un exportateur."""
        if exporter_id in self.exporters:
            del self.exporters[exporter_id]
            logger.info(f"Unregistered exporter {exporter_id}")
            
    async def start_all(self):
        """Démarre tous les exportateurs."""
        for exporter in self.exporters.values():
            if exporter.config.enabled:
                await exporter.start()
        self.running = True
        
    async def stop_all(self):
        """Arrête tous les exportateurs."""
        for exporter in self.exporters.values():
            await exporter.stop()
        self.running = False
        
    async def export_to_all(self, metric: MetricPayload):
        """Exporte une métrique vers tous les exportateurs actifs."""
        tasks = []
        
        for exporter in self.exporters.values():
            if exporter.config.enabled and exporter.is_running:
                tasks.append(exporter.export_metric(metric))
                
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Retourne les stats de tous les exportateurs."""
        stats = {}
        
        for exporter_id, exporter in self.exporters.items():
            stats[exporter_id] = await exporter.get_stats()
            
        return {
            'tenant_id': self.tenant_id,
            'exporters_count': len(self.exporters),
            'running': self.running,
            'exporters': stats
        }


# Factory pour créer des exportateurs personnalisés
class CustomExporterFactory:
    """Factory pour créer des exportateurs personnalisés."""
    
    @staticmethod
    def create_datadog_exporter(
        tenant_id: str,
        api_key: str,
        app_key: str
    ) -> DatadogExporter:
        """Crée un exportateur Datadog."""
        config = ExporterConfig(
            exporter_id=f"datadog_{tenant_id}",
            exporter_type=ExporterType.TIME_SERIES,
            tenant_id=tenant_id,
            batch_size=100,
            retry_attempts=3
        )
        return DatadogExporter(config, api_key, app_key)
        
    @staticmethod
    def create_newrelic_exporter(
        tenant_id: str,
        api_key: str
    ) -> NewRelicExporter:
        """Crée un exportateur New Relic."""
        config = ExporterConfig(
            exporter_id=f"newrelic_{tenant_id}",
            exporter_type=ExporterType.TIME_SERIES,
            tenant_id=tenant_id,
            batch_size=200,
            retry_attempts=3
        )
        return NewRelicExporter(config, api_key)
        
    @staticmethod
    def create_splunk_exporter(
        tenant_id: str,
        hec_url: str,
        hec_token: str
    ) -> SplunkExporter:
        """Crée un exportateur Splunk."""
        config = ExporterConfig(
            exporter_id=f"splunk_{tenant_id}",
            exporter_type=ExporterType.LOG_AGGREGATION,
            tenant_id=tenant_id,
            batch_size=50,
            retry_attempts=2
        )
        return SplunkExporter(config, hec_url, hec_token)
        
    @staticmethod
    def create_webhook_exporter(
        tenant_id: str,
        webhook_url: str,
        auth_token: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> WebhookExporter:
        """Crée un exportateur webhook."""
        config = ExporterConfig(
            exporter_id=f"webhook_{tenant_id}_{uuid.uuid4().hex[:8]}",
            exporter_type=ExporterType.EVENT_STREAMING,
            tenant_id=tenant_id,
            batch_size=10,
            retry_attempts=3
        )
        return WebhookExporter(config, webhook_url, headers, auth_token)


# Middleware communs
async def add_spotify_metadata_middleware(metric: MetricPayload) -> MetricPayload:
    """Ajoute des métadonnées spécifiques à Spotify."""
    metric.metadata.update({
        'platform': 'spotify',
        'service': 'ai-agent',
        'environment': 'production'
    })
    return metric


async def normalize_metric_names_middleware(metric: MetricPayload) -> MetricPayload:
    """Normalise les noms de métriques."""
    metric.metric_name = metric.metric_name.lower().replace(' ', '_').replace('-', '_')
    return metric


async def add_business_context_middleware(metric: MetricPayload) -> MetricPayload:
    """Ajoute le contexte business."""
    if 'artist' in metric.labels:
        metric.metadata['business_unit'] = 'artist_services'
    elif 'collaboration' in metric.metric_name:
        metric.metadata['business_unit'] = 'collaboration_platform'
    else:
        metric.metadata['business_unit'] = 'ai_platform'
    return metric


# Usage example
if __name__ == "__main__":
    async def main():
        # Créer le gestionnaire principal
        exporter_manager = CustomMetricsExporter("spotify_artist_daft_punk")
        
        # Créer des exportateurs
        datadog = CustomExporterFactory.create_datadog_exporter(
            "spotify_artist_daft_punk",
            "your-datadog-api-key",
            "your-datadog-app-key"
        )
        
        webhook = CustomExporterFactory.create_webhook_exporter(
            "spotify_artist_daft_punk",
            "https://your-webhook-url.com/metrics",
            "your-auth-token"
        )
        
        # Ajouter des middleware
        datadog.add_middleware(add_spotify_metadata_middleware)
        datadog.add_middleware(normalize_metric_names_middleware)
        datadog.add_middleware(add_business_context_middleware)
        
        # Enregistrer les exportateurs
        exporter_manager.register_exporter(datadog)
        exporter_manager.register_exporter(webhook)
        
        # Démarrer tous les exportateurs
        await exporter_manager.start_all()
        
        # Exporter une métrique
        metric = MetricPayload(
            tenant_id="spotify_artist_daft_punk",
            metric_name="AI Inference Time",
            metric_value=0.045,
            metric_type="gauge",
            timestamp=datetime.now(),
            labels={
                'model': 'collaborative_filter_v2',
                'artist': 'daft_punk'
            },
            metadata={
                'version': '2.1.0'
            }
        )
        
        await exporter_manager.export_to_all(metric)
        
        # Statistiques
        stats = await exporter_manager.get_all_stats()
        print(f"Exporter stats: {stats}")
        
        # Arrêter
        await exporter_manager.stop_all()
        
    asyncio.run(main())
