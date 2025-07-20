"""
Advanced Batch Metrics Exporter
==============================

Exportateur optimisé pour le traitement par lots de grandes volumes
de métriques avec performance et fiabilité maximales.

Fonctionnalités:
- Traitement par lots haute performance
- Compression automatique
- Partitioning intelligent
- Parallélisation
- Monitoring détaillé
- Recovery automatique
"""

import asyncio
import gzip
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import structlog
from custom_exporter import MetricPayload

logger = structlog.get_logger(__name__)


@dataclass
class BatchConfig:
    """Configuration pour le traitement par lots."""
    batch_size: int = 1000
    max_batch_size: int = 10000
    batch_timeout: float = 30.0
    compression_enabled: bool = True
    compression_level: int = 6
    parallel_workers: int = 4
    memory_limit_mb: int = 512
    disk_buffer_path: str = "/tmp/spotify_ai_metrics"
    retention_hours: int = 24
    partition_by: str = "tenant_id"  # tenant_id, metric_type, timestamp
    enable_recovery: bool = True


@dataclass
class BatchMetrics:
    """Métriques de performance du batch."""
    batch_id: str
    created_at: datetime
    metrics_count: int
    size_bytes: int
    compressed_size_bytes: int
    processing_time_seconds: float
    status: str  # pending, processing, completed, failed
    partition_key: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchMetricsExporter:
    """
    Exportateur optimisé pour le traitement par lots.
    
    Fonctionnalités:
    - Batching intelligent avec compression
    - Partitioning par tenant/type/temps
    - Traitement parallèle
    - Recovery sur échec
    - Monitoring complet
    """
    
    def __init__(self, config: BatchConfig, target_exporters: List[Any]):
        self.config = config
        self.target_exporters = target_exporters
        
        # Buffers par partition
        self.partition_buffers: Dict[str, List[MetricPayload]] = {}
        self.buffer_timestamps: Dict[str, datetime] = {}
        self.buffer_locks: Dict[str, asyncio.Lock] = {}
        
        # Executor pour les tâches CPU-intensives
        self.executor = ThreadPoolExecutor(max_workers=config.parallel_workers)
        
        # Métriques de performance
        self.batch_metrics: List[BatchMetrics] = []
        self.stats = {
            'total_metrics_processed': 0,
            'total_batches_created': 0,
            'total_batches_completed': 0,
            'total_batches_failed': 0,
            'avg_batch_size': 0,
            'avg_processing_time': 0,
            'compression_ratio': 0,
            'memory_usage_mb': 0
        }
        
        # Recovery
        self.failed_batches: List[str] = []
        self.recovery_running = False
        
        # Contrôle du débit
        self.rate_limiter = asyncio.Semaphore(config.parallel_workers)
        
    async def initialize(self):
        """Initialise l'exportateur batch."""
        try:
            # Créer le répertoire de buffer disque
            import os
            os.makedirs(self.config.disk_buffer_path, exist_ok=True)
            
            # Démarrer les tâches de maintenance
            asyncio.create_task(self._batch_timeout_monitor())
            asyncio.create_task(self._memory_monitor())
            
            if self.config.enable_recovery:
                asyncio.create_task(self._recovery_monitor())
                
            # Initialiser les exportateurs cibles
            for exporter in self.target_exporters:
                if hasattr(exporter, 'initialize'):
                    await exporter.initialize()
                    
            logger.info("BatchMetricsExporter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize BatchMetricsExporter: {e}")
            raise
            
    def _get_partition_key(self, metric: MetricPayload) -> str:
        """Génère la clé de partition pour une métrique."""
        if self.config.partition_by == "tenant_id":
            return metric.tenant_id
        elif self.config.partition_by == "metric_type":
            return metric.metric_type
        elif self.config.partition_by == "timestamp":
            # Partition par heure
            return metric.timestamp.strftime("%Y%m%d%H")
        else:
            # Partition combinée
            return f"{metric.tenant_id}_{metric.metric_type}_{metric.timestamp.strftime('%Y%m%d%H')}"
            
    async def add_metric(self, metric: MetricPayload):
        """Ajoute une métrique au buffer approprié."""
        partition_key = self._get_partition_key(metric)
        
        # Initialiser la partition si nécessaire
        if partition_key not in self.partition_buffers:
            self.partition_buffers[partition_key] = []
            self.buffer_timestamps[partition_key] = datetime.now()
            self.buffer_locks[partition_key] = asyncio.Lock()
            
        async with self.buffer_locks[partition_key]:
            self.partition_buffers[partition_key].append(metric)
            
            # Vérifier si on doit créer un batch
            buffer_size = len(self.partition_buffers[partition_key])
            
            if buffer_size >= self.config.batch_size:
                await self._create_batch(partition_key)
                
    async def add_metrics_bulk(self, metrics: List[MetricPayload]):
        """Ajoute plusieurs métriques de manière optimisée."""
        # Grouper par partition
        partitioned_metrics: Dict[str, List[MetricPayload]] = {}
        
        for metric in metrics:
            partition_key = self._get_partition_key(metric)
            if partition_key not in partitioned_metrics:
                partitioned_metrics[partition_key] = []
            partitioned_metrics[partition_key].append(metric)
            
        # Traiter chaque partition
        tasks = []
        for partition_key, partition_metrics in partitioned_metrics.items():
            tasks.append(self._add_metrics_to_partition(partition_key, partition_metrics))
            
        await asyncio.gather(*tasks)
        
    async def _add_metrics_to_partition(
        self, 
        partition_key: str, 
        metrics: List[MetricPayload]
    ):
        """Ajoute des métriques à une partition spécifique."""
        # Initialiser la partition si nécessaire
        if partition_key not in self.partition_buffers:
            self.partition_buffers[partition_key] = []
            self.buffer_timestamps[partition_key] = datetime.now()
            self.buffer_locks[partition_key] = asyncio.Lock()
            
        async with self.buffer_locks[partition_key]:
            self.partition_buffers[partition_key].extend(metrics)
            
            # Créer des batches si nécessaire
            while len(self.partition_buffers[partition_key]) >= self.config.batch_size:
                await self._create_batch(partition_key)
                
    async def _create_batch(self, partition_key: str):
        """Crée un batch à partir du buffer d'une partition."""
        if partition_key not in self.partition_buffers:
            return
            
        buffer = self.partition_buffers[partition_key]
        if not buffer:
            return
            
        # Extraire les métriques pour le batch
        batch_size = min(len(buffer), self.config.max_batch_size)
        batch_metrics = buffer[:batch_size]
        self.partition_buffers[partition_key] = buffer[batch_size:]
        
        # Créer l'ID du batch
        batch_id = hashlib.md5(
            f"{partition_key}_{datetime.now().isoformat()}_{len(batch_metrics)}".encode()
        ).hexdigest()
        
        # Créer les métriques du batch
        batch_info = BatchMetrics(
            batch_id=batch_id,
            created_at=datetime.now(),
            metrics_count=len(batch_metrics),
            size_bytes=0,
            compressed_size_bytes=0,
            processing_time_seconds=0,
            status="pending",
            partition_key=partition_key
        )
        
        self.batch_metrics.append(batch_info)
        self.stats['total_batches_created'] += 1
        
        # Traiter le batch de manière asynchrone
        asyncio.create_task(self._process_batch(batch_id, batch_metrics, batch_info))
        
    async def _process_batch(
        self, 
        batch_id: str, 
        metrics: List[MetricPayload], 
        batch_info: BatchMetrics
    ):
        """Traite un batch de métriques."""
        async with self.rate_limiter:
            start_time = time.time()
            batch_info.status = "processing"
            
            try:
                # Sérialiser les métriques
                serialized_data = await self._serialize_metrics(metrics)
                batch_info.size_bytes = len(serialized_data.encode())
                
                # Compresser si activé
                if self.config.compression_enabled:
                    compressed_data = await self._compress_data(serialized_data)
                    batch_info.compressed_size_bytes = len(compressed_data)
                    data_to_process = compressed_data
                else:
                    data_to_process = serialized_data.encode()
                    batch_info.compressed_size_bytes = batch_info.size_bytes
                    
                # Sauvegarder sur disque pour recovery
                if self.config.enable_recovery:
                    await self._save_batch_to_disk(batch_id, data_to_process)
                    
                # Exporter vers les cibles
                export_success = await self._export_to_targets(metrics, batch_id)
                
                if export_success:
                    batch_info.status = "completed"
                    self.stats['total_batches_completed'] += 1
                    
                    # Nettoyer le fichier de recovery
                    if self.config.enable_recovery:
                        await self._cleanup_batch_file(batch_id)
                else:
                    batch_info.status = "failed"
                    self.stats['total_batches_failed'] += 1
                    self.failed_batches.append(batch_id)
                    
                # Mettre à jour les statistiques
                processing_time = time.time() - start_time
                batch_info.processing_time_seconds = processing_time
                
                self.stats['total_metrics_processed'] += len(metrics)
                self._update_aggregate_stats()
                
                logger.info(
                    f"Processed batch {batch_id}",
                    status=batch_info.status,
                    metrics_count=len(metrics),
                    processing_time=processing_time,
                    compression_ratio=batch_info.size_bytes / max(batch_info.compressed_size_bytes, 1)
                )
                
            except Exception as e:
                batch_info.status = "failed"
                self.stats['total_batches_failed'] += 1
                self.failed_batches.append(batch_id)
                
                logger.error(
                    f"Failed to process batch {batch_id}: {e}",
                    partition_key=batch_info.partition_key,
                    metrics_count=len(metrics)
                )
                
    async def _serialize_metrics(self, metrics: List[MetricPayload]) -> str:
        """Sérialise les métriques en JSON."""
        def serialize_metric(metric: MetricPayload) -> Dict[str, Any]:
            return {
                'tenant_id': metric.tenant_id,
                'metric_name': metric.metric_name,
                'metric_value': metric.metric_value,
                'metric_type': metric.metric_type,
                'timestamp': metric.timestamp.isoformat(),
                'labels': metric.labels,
                'metadata': metric.metadata,
                'source': metric.source
            }
            
        # Utiliser ThreadPoolExecutor pour la sérialisation CPU-intensive
        loop = asyncio.get_event_loop()
        serialized_metrics = await loop.run_in_executor(
            self.executor,
            lambda: [serialize_metric(m) for m in metrics]
        )
        
        return json.dumps({
            'batch_info': {
                'count': len(metrics),
                'created_at': datetime.now().isoformat()
            },
            'metrics': serialized_metrics
        })
        
    async def _compress_data(self, data: str) -> bytes:
        """Compresse les données."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: gzip.compress(
                data.encode('utf-8'), 
                compresslevel=self.config.compression_level
            )
        )
        
    async def _save_batch_to_disk(self, batch_id: str, data: bytes):
        """Sauvegarde un batch sur disque pour recovery."""
        file_path = f"{self.config.disk_buffer_path}/{batch_id}.batch"
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)
            
    async def _cleanup_batch_file(self, batch_id: str):
        """Nettoie le fichier de batch."""
        try:
            import os
            file_path = f"{self.config.disk_buffer_path}/{batch_id}.batch"
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup batch file {batch_id}: {e}")
            
    async def _export_to_targets(self, metrics: List[MetricPayload], batch_id: str) -> bool:
        """Exporte vers tous les exportateurs cibles."""
        export_tasks = []
        
        for exporter in self.target_exporters:
            if hasattr(exporter, 'export_metrics'):
                export_tasks.append(exporter.export_metrics(metrics))
            else:
                # Exporter une par une si pas de support batch
                for metric in metrics:
                    export_tasks.append(exporter.export_metric(metric))
                    
        # Exécuter tous les exports
        try:
            results = await asyncio.gather(*export_tasks, return_exceptions=True)
            
            # Vérifier les résultats
            failures = sum(1 for result in results if isinstance(result, Exception))
            success_rate = (len(results) - failures) / len(results) if results else 0
            
            # Considérer comme succès si au moins 50% des exports réussissent
            return success_rate >= 0.5
            
        except Exception as e:
            logger.error(f"Export failed for batch {batch_id}: {e}")
            return False
            
    async def _batch_timeout_monitor(self):
        """Surveille les timeouts des batches."""
        while True:
            try:
                current_time = datetime.now()
                expired_partitions = []
                
                for partition_key, timestamp in self.buffer_timestamps.items():
                    if (current_time - timestamp).total_seconds() > self.config.batch_timeout:
                        expired_partitions.append(partition_key)
                        
                # Forcer la création de batches pour les partitions expirées
                for partition_key in expired_partitions:
                    if partition_key in self.partition_buffers and self.partition_buffers[partition_key]:
                        async with self.buffer_locks[partition_key]:
                            await self._create_batch(partition_key)
                            self.buffer_timestamps[partition_key] = current_time
                            
                await asyncio.sleep(10)  # Vérifier toutes les 10 secondes
                
            except Exception as e:
                logger.error(f"Batch timeout monitor error: {e}")
                await asyncio.sleep(30)
                
    async def _memory_monitor(self):
        """Surveille l'utilisation mémoire."""
        while True:
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                self.stats['memory_usage_mb'] = memory_mb
                
                # Si mémoire excessive, forcer des batches
                if memory_mb > self.config.memory_limit_mb:
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB, forcing batch creation")
                    
                    for partition_key in list(self.partition_buffers.keys()):
                        if self.partition_buffers[partition_key]:
                            async with self.buffer_locks[partition_key]:
                                await self._create_batch(partition_key)
                                
                await asyncio.sleep(30)  # Vérifier toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                await asyncio.sleep(60)
                
    async def _recovery_monitor(self):
        """Surveille et récupère les batches échoués."""
        while True:
            try:
                if self.failed_batches and not self.recovery_running:
                    self.recovery_running = True
                    
                    failed_batch_id = self.failed_batches.pop(0)
                    await self._recover_batch(failed_batch_id)
                    
                    self.recovery_running = False
                    
                await asyncio.sleep(60)  # Vérifier toutes les minutes
                
            except Exception as e:
                logger.error(f"Recovery monitor error: {e}")
                self.recovery_running = False
                await asyncio.sleep(120)
                
    async def _recover_batch(self, batch_id: str):
        """Récupère un batch échoué."""
        try:
            file_path = f"{self.config.disk_buffer_path}/{batch_id}.batch"
            
            import os
            if not os.path.exists(file_path):
                logger.warning(f"Recovery file not found for batch {batch_id}")
                return
                
            # Charger les données
            async with aiofiles.open(file_path, 'rb') as f:
                compressed_data = await f.read()
                
            # Décompresser si nécessaire
            if self.config.compression_enabled:
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    self.executor,
                    lambda: gzip.decompress(compressed_data).decode('utf-8')
                )
            else:
                data = compressed_data.decode('utf-8')
                
            # Désérialiser
            batch_data = json.loads(data)
            metrics = []
            
            for metric_data in batch_data['metrics']:
                metric = MetricPayload(
                    tenant_id=metric_data['tenant_id'],
                    metric_name=metric_data['metric_name'],
                    metric_value=metric_data['metric_value'],
                    metric_type=metric_data['metric_type'],
                    timestamp=datetime.fromisoformat(metric_data['timestamp']),
                    labels=metric_data['labels'],
                    metadata=metric_data['metadata'],
                    source=metric_data['source']
                )
                metrics.append(metric)
                
            # Réessayer l'export
            success = await self._export_to_targets(metrics, batch_id)
            
            if success:
                await self._cleanup_batch_file(batch_id)
                logger.info(f"Successfully recovered batch {batch_id}")
            else:
                # Remettre dans la queue pour retry plus tard
                self.failed_batches.append(batch_id)
                logger.warning(f"Recovery failed for batch {batch_id}, will retry later")
                
        except Exception as e:
            logger.error(f"Failed to recover batch {batch_id}: {e}")
            
    def _update_aggregate_stats(self):
        """Met à jour les statistiques agrégées."""
        if self.batch_metrics:
            completed_batches = [b for b in self.batch_metrics if b.status == "completed"]
            
            if completed_batches:
                self.stats['avg_batch_size'] = sum(b.metrics_count for b in completed_batches) / len(completed_batches)
                self.stats['avg_processing_time'] = sum(b.processing_time_seconds for b in completed_batches) / len(completed_batches)
                
                # Ratio de compression
                total_original = sum(b.size_bytes for b in completed_batches)
                total_compressed = sum(b.compressed_size_bytes for b in completed_batches)
                
                if total_compressed > 0:
                    self.stats['compression_ratio'] = total_original / total_compressed
                    
    async def flush_all_buffers(self):
        """Force le vidage de tous les buffers."""
        tasks = []
        
        for partition_key in list(self.partition_buffers.keys()):
            if self.partition_buffers[partition_key]:
                tasks.append(self._flush_partition_buffer(partition_key))
                
        if tasks:
            await asyncio.gather(*tasks)
            
    async def _flush_partition_buffer(self, partition_key: str):
        """Vide le buffer d'une partition."""
        async with self.buffer_locks[partition_key]:
            while self.partition_buffers[partition_key]:
                await self._create_batch(partition_key)
                
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de performance détaillées."""
        recent_batches = [
            b for b in self.batch_metrics 
            if (datetime.now() - b.created_at).total_seconds() < 3600  # Dernière heure
        ]
        
        return {
            'overall_stats': self.stats,
            'recent_performance': {
                'batches_last_hour': len(recent_batches),
                'avg_processing_time_last_hour': sum(b.processing_time_seconds for b in recent_batches) / max(len(recent_batches), 1),
                'success_rate_last_hour': len([b for b in recent_batches if b.status == "completed"]) / max(len(recent_batches), 1)
            },
            'buffer_status': {
                'active_partitions': len(self.partition_buffers),
                'total_buffered_metrics': sum(len(buffer) for buffer in self.partition_buffers.values()),
                'pending_batches': len([b for b in self.batch_metrics if b.status == "pending"]),
                'failed_batches': len(self.failed_batches)
            },
            'resource_usage': {
                'memory_usage_mb': self.stats['memory_usage_mb'],
                'disk_usage_gb': await self._get_disk_usage()
            }
        }
        
    async def _get_disk_usage(self) -> float:
        """Calcule l'utilisation disque du buffer."""
        try:
            import os
            total_size = 0
            for filename in os.listdir(self.config.disk_buffer_path):
                if filename.endswith('.batch'):
                    filepath = os.path.join(self.config.disk_buffer_path, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024 * 1024)  # GB
        except Exception:
            return 0.0
            
    async def cleanup(self):
        """Nettoie les ressources."""
        # Vider tous les buffers
        await self.flush_all_buffers()
        
        # Attendre que tous les batches en cours se terminent
        await asyncio.sleep(5)
        
        # Fermer l'executor
        self.executor.shutdown(wait=True)
        
        logger.info("BatchMetricsExporter cleaned up")


# Usage example
if __name__ == "__main__":
    async def main():
        from prometheus_exporter import PrometheusExporterFactory
        from elastic_exporter import ElasticsearchExporterFactory
        
        # Créer des exportateurs cibles
        prometheus_exporter = PrometheusExporterFactory.create_spotify_ai_exporter(
            "spotify_artist_daft_punk"
        )
        
        elastic_exporter = ElasticsearchExporterFactory.create_spotify_ai_exporter(
            "spotify_artist_daft_punk"
        )
        
        # Configuration batch
        batch_config = BatchConfig(
            batch_size=500,
            max_batch_size=2000,
            batch_timeout=30.0,
            compression_enabled=True,
            parallel_workers=6,
            memory_limit_mb=1024
        )
        
        # Créer l'exportateur batch
        batch_exporter = BatchMetricsExporter(
            batch_config,
            [prometheus_exporter, elastic_exporter]
        )
        
        await batch_exporter.initialize()
        
        # Simuler des métriques
        metrics = []
        for i in range(5000):
            metric = MetricPayload(
                tenant_id="spotify_artist_daft_punk",
                metric_name=f"test_metric_{i % 10}",
                metric_value=i * 0.1,
                metric_type="gauge",
                timestamp=datetime.now(),
                labels={'batch': 'test', 'sequence': str(i)}
            )
            metrics.append(metric)
            
        # Ajouter en lot
        await batch_exporter.add_metrics_bulk(metrics)
        
        # Attendre le traitement
        await asyncio.sleep(10)
        
        # Statistiques
        stats = await batch_exporter.get_performance_stats()
        print(f"Performance stats: {stats}")
        
        await batch_exporter.cleanup()
        
    asyncio.run(main())
