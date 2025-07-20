"""
Advanced Prometheus Multi-Tenant Metrics Exporter
================================================

Exportateur haute performance pour métriques Prometheus avec isolation complète
des tenants dans l'environnement Spotify AI Agent.

Fonctionnalités:
- Isolation tenant native
- Chiffrement AES-256 des métriques
- Compression GZIP automatique
- Rate limiting intelligent
- Retry avec backoff exponentiel
- Circuit breaker pattern
"""

import asyncio
import time
import gzip
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest
from prometheus_client.exposition import MetricsHandler
from cryptography.fernet import Fernet
import aioredis
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TenantMetricConfig:
    """Configuration des métriques par tenant."""
    tenant_id: str
    namespace: str = "spotify_ai"
    encryption_enabled: bool = True
    compression_level: int = 6
    rate_limit_per_second: int = 1000
    batch_size: int = 100
    retention_days: int = 30
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricData:
    """Structure de données pour une métrique."""
    name: str
    value: float
    metric_type: str  # gauge, counter, histogram
    labels: Dict[str, str]
    timestamp: datetime
    tenant_id: str
    help_text: str = ""


class PrometheusMultiTenantExporter:
    """
    Exportateur Prometheus avancé avec support multi-tenant.
    
    Fonctionnalités:
    - Isolation complète des tenants
    - Chiffrement end-to-end
    - Performance optimisée
    - Monitoring self-service
    """
    
    def __init__(
        self,
        tenant_config: TenantMetricConfig,
        prometheus_url: str = "http://localhost:9090",
        redis_url: str = "redis://localhost:6379",
        encryption_key: Optional[str] = None
    ):
        self.config = tenant_config
        self.prometheus_url = prometheus_url
        self.redis_url = redis_url
        
        # Chiffrement
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode())
        else:
            self.cipher = Fernet(Fernet.generate_key())
            
        # Registres Prometheus par tenant
        self.registries: Dict[str, CollectorRegistry] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.rate_limiter = {}
        self.last_reset = time.time()
        
        # Circuit breaker
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': None,
            'state': 'closed'  # closed, open, half-open
        }
        
        # Cache Redis
        self.redis_client = None
        
        # Métriques internes
        self._setup_internal_metrics()
        
    async def initialize(self):
        """Initialise l'exportateur."""
        try:
            # Connexion Redis
            self.redis_client = await aioredis.from_url(self.redis_url)
            
            # Setup tenant registry
            await self._setup_tenant_registry()
            
            # Warmup cache
            await self._warmup_cache()
            
            logger.info(
                "PrometheusMultiTenantExporter initialized",
                tenant_id=self.config.tenant_id
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize exporter",
                error=str(e),
                tenant_id=self.config.tenant_id
            )
            raise
            
    def _setup_internal_metrics(self):
        """Configure les métriques internes de monitoring."""
        self.internal_registry = CollectorRegistry()
        
        # Métriques de performance
        self.export_duration = Histogram(
            'spotify_ai_export_duration_seconds',
            'Durée d\'exportation des métriques',
            ['tenant_id', 'export_type'],
            registry=self.internal_registry
        )
        
        self.export_counter = Counter(
            'spotify_ai_exports_total',
            'Nombre total d\'exportations',
            ['tenant_id', 'status'],
            registry=self.internal_registry
        )
        
        self.rate_limit_hits = Counter(
            'spotify_ai_rate_limit_hits_total',
            'Nombre de hits de rate limiting',
            ['tenant_id'],
            registry=self.internal_registry
        )
        
        self.circuit_breaker_state = Gauge(
            'spotify_ai_circuit_breaker_state',
            'État du circuit breaker (0=closed, 1=open, 2=half-open)',
            ['tenant_id'],
            registry=self.internal_registry
        )
        
    async def _setup_tenant_registry(self):
        """Configure le registre Prometheus pour le tenant."""
        tenant_id = self.config.tenant_id
        
        if tenant_id not in self.registries:
            self.registries[tenant_id] = CollectorRegistry()
            self.metrics[tenant_id] = {}
            
        # Métriques business Spotify AI
        registry = self.registries[tenant_id]
        
        # Métriques IA
        self.metrics[tenant_id]['ai_inference_time'] = Histogram(
            f'{self.config.namespace}_ai_inference_duration_seconds',
            'Temps d\'inférence des modèles IA',
            ['model_name', 'model_version', 'tensor_size'],
            registry=registry
        )
        
        self.metrics[tenant_id]['recommendation_accuracy'] = Gauge(
            f'{self.config.namespace}_recommendation_accuracy_ratio',
            'Précision des recommandations IA',
            ['algorithm', 'dataset_version'],
            registry=registry
        )
        
        # Métriques Spotify Business
        self.metrics[tenant_id]['tracks_generated'] = Counter(
            f'{self.config.namespace}_tracks_generated_total',
            'Nombre de pistes générées',
            ['genre', 'collaboration_type'],
            registry=registry
        )
        
        self.metrics[tenant_id]['artist_engagement'] = Gauge(
            f'{self.config.namespace}_artist_engagement_score',
            'Score d\'engagement des artistes',
            ['artist_tier', 'region'],
            registry=registry
        )
        
        self.metrics[tenant_id]['revenue_impact'] = Gauge(
            f'{self.config.namespace}_revenue_impact_euros',
            'Impact revenue en euros',
            ['stream_type', 'monetization_channel'],
            registry=registry
        )
        
        # Métriques collaboratives
        self.metrics[tenant_id]['collaboration_success'] = Histogram(
            f'{self.config.namespace}_collaboration_success_rate',
            'Taux de succès des collaborations',
            ['collaboration_type', 'genre_match'],
            registry=registry
        )
        
    async def _warmup_cache(self):
        """Précharge le cache avec les configurations."""
        cache_key = f"tenant_config:{self.config.tenant_id}"
        config_data = {
            'namespace': self.config.namespace,
            'rate_limit': self.config.rate_limit_per_second,
            'batch_size': self.config.batch_size,
            'labels': self.config.labels
        }
        
        await self.redis_client.setex(
            cache_key,
            timedelta(hours=1),
            json.dumps(config_data)
        )
        
    def _check_rate_limit(self) -> bool:
        """Vérifie les limites de taux."""
        current_time = time.time()
        tenant_id = self.config.tenant_id
        
        # Reset toutes les secondes
        if current_time - self.last_reset >= 1.0:
            self.rate_limiter = {}
            self.last_reset = current_time
            
        # Vérifier le tenant
        current_count = self.rate_limiter.get(tenant_id, 0)
        
        if current_count >= self.config.rate_limit_per_second:
            self.rate_limit_hits.labels(tenant_id=tenant_id).inc()
            return False
            
        self.rate_limiter[tenant_id] = current_count + 1
        return True
        
    def _check_circuit_breaker(self) -> bool:
        """Vérifie l'état du circuit breaker."""
        current_time = time.time()
        
        if self.circuit_breaker['state'] == 'open':
            # Vérifier si on peut passer en half-open
            if (current_time - self.circuit_breaker['last_failure']) > 60:
                self.circuit_breaker['state'] = 'half-open'
                self.circuit_breaker_state.labels(
                    tenant_id=self.config.tenant_id
                ).set(2)
                return True
            return False
            
        return True
        
    def _record_success(self):
        """Enregistre un succès pour le circuit breaker."""
        if self.circuit_breaker['state'] == 'half-open':
            self.circuit_breaker['state'] = 'closed'
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker_state.labels(
                tenant_id=self.config.tenant_id
            ).set(0)
            
    def _record_failure(self):
        """Enregistre un échec pour le circuit breaker."""
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= 5:
            self.circuit_breaker['state'] = 'open'
            self.circuit_breaker_state.labels(
                tenant_id=self.config.tenant_id
            ).set(1)
            
    def _encrypt_metric(self, data: Dict[str, Any]) -> str:
        """Chiffre les données de métrique."""
        if not self.config.encryption_enabled:
            return json.dumps(data)
            
        json_data = json.dumps(data)
        encrypted = self.cipher.encrypt(json_data.encode())
        return encrypted.decode()
        
    def _compress_data(self, data: str) -> bytes:
        """Compresse les données."""
        return gzip.compress(
            data.encode(),
            compresslevel=self.config.compression_level
        )
        
    async def export_ai_metrics(self, metrics_data: Dict[str, Any]):
        """
        Exporte les métriques IA vers Prometheus.
        
        Args:
            metrics_data: Dictionnaire contenant les métriques IA
        """
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
            
        if not self._check_circuit_breaker():
            raise Exception("Circuit breaker is open")
            
        start_time = time.time()
        
        try:
            tenant_metrics = self.metrics[self.config.tenant_id]
            
            # Métriques d'inférence
            if 'inference_time' in metrics_data:
                tenant_metrics['ai_inference_time'].labels(
                    model_name=metrics_data.get('model_name', 'unknown'),
                    model_version=metrics_data.get('model_version', 'v1.0'),
                    tensor_size=metrics_data.get('tensor_size', 'medium')
                ).observe(metrics_data['inference_time'])
                
            # Précision des recommandations
            if 'recommendation_accuracy' in metrics_data:
                tenant_metrics['recommendation_accuracy'].labels(
                    algorithm=metrics_data.get('algorithm', 'collaborative'),
                    dataset_version=metrics_data.get('dataset_version', 'latest')
                ).set(metrics_data['recommendation_accuracy'])
                
            # Engagement utilisateur
            if 'user_engagement' in metrics_data:
                tenant_metrics['artist_engagement'].labels(
                    artist_tier=metrics_data.get('artist_tier', 'emerging'),
                    region=metrics_data.get('region', 'global')
                ).set(metrics_data['user_engagement'])
                
            await self._push_to_prometheus()
            
            self._record_success()
            self.export_counter.labels(
                tenant_id=self.config.tenant_id,
                status='success'
            ).inc()
            
        except Exception as e:
            self._record_failure()
            self.export_counter.labels(
                tenant_id=self.config.tenant_id,
                status='error'
            ).inc()
            raise
            
        finally:
            duration = time.time() - start_time
            self.export_duration.labels(
                tenant_id=self.config.tenant_id,
                export_type='ai_metrics'
            ).observe(duration)
            
    async def export_business_metrics(self, metrics_data: Dict[str, Any]):
        """
        Exporte les métriques business Spotify.
        
        Args:
            metrics_data: Dictionnaire contenant les métriques business
        """
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
            
        start_time = time.time()
        
        try:
            tenant_metrics = self.metrics[self.config.tenant_id]
            
            # Pistes générées
            if 'tracks_generated' in metrics_data:
                tenant_metrics['tracks_generated'].labels(
                    genre=metrics_data.get('genre', 'pop'),
                    collaboration_type=metrics_data.get('collaboration_type', 'single')
                ).inc(metrics_data['tracks_generated'])
                
            # Impact revenue
            if 'revenue_impact' in metrics_data:
                tenant_metrics['revenue_impact'].labels(
                    stream_type=metrics_data.get('stream_type', 'premium'),
                    monetization_channel=metrics_data.get('channel', 'streaming')
                ).set(metrics_data['revenue_impact'])
                
            # Succès collaborations
            if 'collaboration_success_rate' in metrics_data:
                tenant_metrics['collaboration_success'].labels(
                    collaboration_type=metrics_data.get('collaboration_type', 'ai_assisted'),
                    genre_match=metrics_data.get('genre_match', 'high')
                ).observe(metrics_data['collaboration_success_rate'])
                
            await self._push_to_prometheus()
            
            self.export_counter.labels(
                tenant_id=self.config.tenant_id,
                status='success'
            ).inc()
            
        except Exception as e:
            self.export_counter.labels(
                tenant_id=self.config.tenant_id,
                status='error'
            ).inc()
            raise
            
        finally:
            duration = time.time() - start_time
            self.export_duration.labels(
                tenant_id=self.config.tenant_id,
                export_type='business_metrics'
            ).observe(duration)
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _push_to_prometheus(self):
        """Pousse les métriques vers Prometheus."""
        registry = self.registries[self.config.tenant_id]
        
        # Génération du format Prometheus
        metrics_output = generate_latest(registry).decode('utf-8')
        
        # Chiffrement si activé
        if self.config.encryption_enabled:
            metrics_output = self._encrypt_metric({'data': metrics_output})
            
        # Compression
        compressed_data = self._compress_data(metrics_output)
        
        # Push vers Prometheus Gateway
        pushgateway_url = f"{self.prometheus_url}/metrics/job/spotify_ai_agent/instance/{self.config.tenant_id}"
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'Content-Type': 'application/x-protobuf',
                'Content-Encoding': 'gzip',
                'X-Tenant-ID': self.config.tenant_id
            }
            
            async with session.post(
                pushgateway_url,
                data=compressed_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status >= 400:
                    raise Exception(f"Prometheus push failed: {response.status}")
                    
    async def export_custom_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: str = 'gauge',
        labels: Optional[Dict[str, str]] = None,
        help_text: str = ""
    ):
        """
        Exporte une métrique personnalisée.
        
        Args:
            metric_name: Nom de la métrique
            value: Valeur de la métrique
            metric_type: Type (gauge, counter, histogram)
            labels: Labels additionnels
            help_text: Texte d'aide
        """
        registry = self.registries[self.config.tenant_id]
        full_name = f"{self.config.namespace}_{metric_name}"
        
        if full_name not in self.metrics[self.config.tenant_id]:
            if metric_type == 'gauge':
                metric = Gauge(full_name, help_text, labels.keys() if labels else [], registry=registry)
            elif metric_type == 'counter':
                metric = Counter(full_name, help_text, labels.keys() if labels else [], registry=registry)
            elif metric_type == 'histogram':
                metric = Histogram(full_name, help_text, labels.keys() if labels else [], registry=registry)
            else:
                raise ValueError(f"Unsupported metric type: {metric_type}")
                
            self.metrics[self.config.tenant_id][full_name] = metric
            
        metric = self.metrics[self.config.tenant_id][full_name]
        
        if labels:
            metric = metric.labels(**labels)
            
        if metric_type == 'gauge':
            metric.set(value)
        elif metric_type == 'counter':
            metric.inc(value)
        elif metric_type == 'histogram':
            metric.observe(value)
            
        await self._push_to_prometheus()
        
    async def get_tenant_metrics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques du tenant."""
        cache_key = f"metrics_summary:{self.config.tenant_id}"
        
        # Vérifier le cache
        cached = await self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
            
        # Calculer le résumé
        registry = self.registries[self.config.tenant_id]
        metrics_output = generate_latest(registry).decode('utf-8')
        
        summary = {
            'tenant_id': self.config.tenant_id,
            'metrics_count': len(self.metrics[self.config.tenant_id]),
            'last_export': datetime.now().isoformat(),
            'rate_limit_status': {
                'current': self.rate_limiter.get(self.config.tenant_id, 0),
                'limit': self.config.rate_limit_per_second
            },
            'circuit_breaker_state': self.circuit_breaker['state'],
            'data_size_bytes': len(metrics_output.encode())
        }
        
        # Cache pendant 5 minutes
        await self.redis_client.setex(
            cache_key,
            timedelta(minutes=5),
            json.dumps(summary)
        )
        
        return summary
        
    async def cleanup(self):
        """Nettoie les ressources."""
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info(
            "PrometheusMultiTenantExporter cleaned up",
            tenant_id=self.config.tenant_id
        )


# Factory pour créer des exportateurs
class PrometheusExporterFactory:
    """Factory pour créer des exportateurs Prometheus configurés."""
    
    @staticmethod
    def create_spotify_ai_exporter(
        tenant_id: str,
        custom_labels: Optional[Dict[str, str]] = None
    ) -> PrometheusMultiTenantExporter:
        """Crée un exportateur configuré pour Spotify AI."""
        labels = {
            'service': 'spotify-ai-agent',
            'environment': 'production',
            'team': 'ai-platform'
        }
        
        if custom_labels:
            labels.update(custom_labels)
            
        config = TenantMetricConfig(
            tenant_id=tenant_id,
            namespace="spotify_ai",
            encryption_enabled=True,
            compression_level=6,
            rate_limit_per_second=2000,
            batch_size=500,
            labels=labels
        )
        
        return PrometheusMultiTenantExporter(config)


# Usage example
if __name__ == "__main__":
    async def main():
        # Configuration pour un artiste Spotify
        exporter = PrometheusExporterFactory.create_spotify_ai_exporter(
            tenant_id="spotify_artist_daft_punk",
            custom_labels={'genre': 'electronic', 'region': 'eu'}
        )
        
        await exporter.initialize()
        
        # Export de métriques IA
        await exporter.export_ai_metrics({
            'inference_time': 0.045,
            'recommendation_accuracy': 0.94,
            'user_engagement': 8.7,
            'model_name': 'collaborative_filter_v2',
            'algorithm': 'deep_learning'
        })
        
        # Export de métriques business
        await exporter.export_business_metrics({
            'tracks_generated': 25,
            'revenue_impact': 15000.50,
            'collaboration_success_rate': 0.89,
            'genre': 'electronic',
            'collaboration_type': 'ai_assisted'
        })
        
        # Résumé des métriques
        summary = await exporter.get_tenant_metrics_summary()
        print(f"Metrics summary: {summary}")
        
        await exporter.cleanup()
        
    asyncio.run(main())
