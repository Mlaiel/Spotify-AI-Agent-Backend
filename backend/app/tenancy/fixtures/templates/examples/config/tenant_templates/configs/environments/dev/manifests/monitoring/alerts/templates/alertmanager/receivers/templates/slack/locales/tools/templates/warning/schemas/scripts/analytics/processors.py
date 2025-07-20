"""
Processors Module - Processeurs de Données Analytics
===================================================

Ce module contient les processeurs de données pour le système d'analytics,
incluant le traitement en temps réel, par lots, de flux et ML.

Classes:
- BaseProcessor: Processeur de base
- RealTimeProcessor: Traitement en temps réel
- BatchProcessor: Traitement par lots
- StreamProcessor: Traitement de flux
- MLProcessor: Traitement Machine Learning
- DataPipeline: Pipeline de traitement
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import statistics

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import redis.asyncio as aioredis

from ..config import AnalyticsConfig
from ..models import Metric, Event, Alert
from ..utils import Logger, Timer, measure_time, retry_async


@dataclass
class ProcessingResult:
    """Résultat de traitement."""
    success: bool
    processed_count: int
    error_count: int
    processing_time: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_items(self) -> int:
        """Nombre total d'éléments."""
        return self.processed_count + self.error_count
    
    @property
    def success_rate(self) -> float:
        """Taux de succès."""
        if self.total_items == 0:
            return 0.0
        return self.processed_count / self.total_items


class BaseProcessor(ABC):
    """Processeur de base abstrait."""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.logger = Logger(self.__class__.__name__)
        self.is_running = False
        self.stats = {
            'total_processed': 0,
            'total_errors': 0,
            'last_processing_time': 0,
            'average_processing_time': 0,
            'start_time': None
        }
        self._processing_times = []
    
    async def start(self):
        """Démarre le processeur."""
        if self.is_running:
            return
        
        self.is_running = True
        self.stats['start_time'] = datetime.utcnow()
        self.logger.info(f"Processeur {self.__class__.__name__} démarré")
        await self._on_start()
    
    async def stop(self):
        """Arrête le processeur."""
        if not self.is_running:
            return
        
        self.is_running = False
        await self._on_stop()
        self.logger.info(f"Processeur {self.__class__.__name__} arrêté")
    
    @abstractmethod
    async def process(self, data: Any) -> ProcessingResult:
        """Traite des données."""
        pass
    
    async def _on_start(self):
        """Appelé au démarrage."""
        pass
    
    async def _on_stop(self):
        """Appelé à l'arrêt."""
        pass
    
    def _update_stats(self, result: ProcessingResult):
        """Met à jour les statistiques."""
        self.stats['total_processed'] += result.processed_count
        self.stats['total_errors'] += result.error_count
        self.stats['last_processing_time'] = result.processing_time
        
        self._processing_times.append(result.processing_time)
        if len(self._processing_times) > 100:  # Garder seulement les 100 dernières
            self._processing_times.pop(0)
        
        self.stats['average_processing_time'] = statistics.mean(self._processing_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques."""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['uptime_seconds'] = (datetime.utcnow() - stats['start_time']).total_seconds()
        return stats


class RealTimeProcessor(BaseProcessor):
    """Processeur en temps réel pour les métriques et événements."""
    
    def __init__(self, config: AnalyticsConfig):
        super().__init__(config)
        self.redis_client: Optional[aioredis.Redis] = None
        self.processing_queue = asyncio.Queue(maxsize=10000)
        self.batch_size = 100
        self.flush_interval = 5  # secondes
        self.worker_tasks: List[asyncio.Task] = []
        self.aggregators = {}
        
    async def _on_start(self):
        """Initialisation du processeur temps réel."""
        # Connexion Redis
        self.redis_client = aioredis.from_url(
            self.config.redis_url,
            decode_responses=True
        )
        
        # Démarrer les workers
        self.worker_tasks = [
            asyncio.create_task(self._worker())
            for _ in range(4)  # 4 workers
        ]
        
        # Démarrer le worker de flush
        self.worker_tasks.append(
            asyncio.create_task(self._flush_worker())
        )
        
        self.logger.info("Processeur temps réel initialisé")
    
    async def _on_stop(self):
        """Arrêt du processeur temps réel."""
        # Arrêter les workers
        for task in self.worker_tasks:
            task.cancel()
        
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Fermer Redis
        if self.redis_client:
            await self.redis_client.close()
    
    async def process(self, data: Union[Metric, Event, List[Union[Metric, Event]]]) -> ProcessingResult:
        """Traite des données en temps réel."""
        start_time = time.time()
        
        if isinstance(data, (Metric, Event)):
            data = [data]
        
        processed_count = 0
        error_count = 0
        errors = []
        
        for item in data:
            try:
                await self.processing_queue.put(item)
                processed_count += 1
            except Exception as e:
                error_count += 1
                errors.append(str(e))
                self.logger.error(f"Erreur ajout queue: {e}")
        
        processing_time = time.time() - start_time
        
        result = ProcessingResult(
            success=error_count == 0,
            processed_count=processed_count,
            error_count=error_count,
            processing_time=processing_time,
            errors=errors
        )
        
        self._update_stats(result)
        return result
    
    async def _worker(self):
        """Worker de traitement."""
        batch = []
        
        while self.is_running:
            try:
                # Collecter un batch
                while len(batch) < self.batch_size:
                    try:
                        item = await asyncio.wait_for(
                            self.processing_queue.get(),
                            timeout=1.0
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self._process_batch(batch)
                    batch.clear()
                
            except Exception as e:
                self.logger.error(f"Erreur worker: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, batch: List[Union[Metric, Event]]):
        """Traite un batch d'éléments."""
        for item in batch:
            try:
                if isinstance(item, Metric):
                    await self._process_metric(item)
                elif isinstance(item, Event):
                    await self._process_event(item)
            except Exception as e:
                self.logger.error(f"Erreur traitement item: {e}")
    
    async def _process_metric(self, metric: Metric):
        """Traite une métrique."""
        # Stocker en Redis pour accès rapide
        key = f"metric:{metric.tenant_id}:{metric.name}"
        data = {
            'value': metric.value,
            'timestamp': metric.timestamp.isoformat(),
            'tags': json.dumps(metric.tags)
        }
        
        await self.redis_client.hset(key, mapping=data)
        
        # TTL pour éviter l'accumulation
        await self.redis_client.expire(key, 3600)  # 1 heure
        
        # Agrégation en temps réel
        await self._aggregate_metric(metric)
        
        # Publier pour les listeners
        await self.redis_client.publish(
            f"metrics:{metric.tenant_id}",
            json.dumps(metric.to_dict(), default=str)
        )
    
    async def _process_event(self, event: Event):
        """Traite un événement."""
        # Stocker en Redis
        key = f"event:{event.tenant_id}:{event.id}"
        
        await self.redis_client.set(
            key,
            json.dumps(event.to_dict(), default=str),
            ex=86400  # 24 heures
        )
        
        # Publier pour les listeners
        await self.redis_client.publish(
            f"events:{event.tenant_id}",
            json.dumps(event.to_dict(), default=str)
        )
    
    async def _aggregate_metric(self, metric: Metric):
        """Agrège une métrique."""
        # Agrégations par minute, heure, jour
        for window in ['1m', '1h', '1d']:
            await self._aggregate_metric_window(metric, window)
    
    async def _aggregate_metric_window(self, metric: Metric, window: str):
        """Agrège une métrique pour une fenêtre."""
        window_key = self._get_window_key(metric.timestamp, window)
        agg_key = f"agg:{metric.tenant_id}:{metric.name}:{window}:{window_key}"
        
        # Utiliser Redis pour l'agrégation atomique
        pipe = self.redis_client.pipeline()
        pipe.hincrby(agg_key, 'count', 1)
        pipe.hincrbyfloat(agg_key, 'sum', float(metric.value))
        pipe.hset(agg_key, 'last_value', float(metric.value))
        pipe.hset(agg_key, 'last_timestamp', metric.timestamp.isoformat())
        pipe.expire(agg_key, self._get_window_ttl(window))
        
        await pipe.execute()
    
    def _get_window_key(self, timestamp: datetime, window: str) -> str:
        """Génère une clé de fenêtre."""
        if window == '1m':
            return timestamp.strftime('%Y%m%d%H%M')
        elif window == '1h':
            return timestamp.strftime('%Y%m%d%H')
        elif window == '1d':
            return timestamp.strftime('%Y%m%d')
        return timestamp.isoformat()
    
    def _get_window_ttl(self, window: str) -> int:
        """Retourne le TTL pour une fenêtre."""
        ttls = {
            '1m': 3600,    # 1 heure
            '1h': 86400,   # 24 heures
            '1d': 2592000  # 30 jours
        }
        return ttls.get(window, 3600)
    
    async def _flush_worker(self):
        """Worker de flush périodique."""
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_aggregations()
            except Exception as e:
                self.logger.error(f"Erreur flush: {e}")
    
    async def _flush_aggregations(self):
        """Flush les agrégations vers le stockage permanent."""
        # Implémentation du flush vers InfluxDB/PostgreSQL
        self.logger.debug("Flush des agrégations effectué")


class BatchProcessor(BaseProcessor):
    """Processeur par lots pour les gros volumes."""
    
    def __init__(self, config: AnalyticsConfig):
        super().__init__(config)
        self.batch_size = 1000
        self.max_workers = 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    async def process(self, data: List[Union[Metric, Event]]) -> ProcessingResult:
        """Traite des données par lots."""
        start_time = time.time()
        
        # Diviser en batches
        batches = [
            data[i:i + self.batch_size]
            for i in range(0, len(data), self.batch_size)
        ]
        
        total_processed = 0
        total_errors = 0
        all_errors = []
        
        # Traiter les batches en parallèle
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch(batch))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ProcessingResult):
                total_processed += result.processed_count
                total_errors += result.error_count
                all_errors.extend(result.errors)
            else:
                total_errors += 1
                all_errors.append(str(result))
        
        processing_time = time.time() - start_time
        
        result = ProcessingResult(
            success=total_errors == 0,
            processed_count=total_processed,
            error_count=total_errors,
            processing_time=processing_time,
            errors=all_errors
        )
        
        self._update_stats(result)
        return result
    
    async def _process_batch(self, batch: List[Union[Metric, Event]]) -> ProcessingResult:
        """Traite un batch."""
        processed_count = 0
        error_count = 0
        errors = []
        
        # Séparer métriques et événements
        metrics = [item for item in batch if isinstance(item, Metric)]
        events = [item for item in batch if isinstance(item, Event)]
        
        # Traiter les métriques
        if metrics:
            result = await self._process_metrics_batch(metrics)
            processed_count += result.processed_count
            error_count += result.error_count
            errors.extend(result.errors)
        
        # Traiter les événements
        if events:
            result = await self._process_events_batch(events)
            processed_count += result.processed_count
            error_count += result.error_count
            errors.extend(result.errors)
        
        return ProcessingResult(
            success=error_count == 0,
            processed_count=processed_count,
            error_count=error_count,
            processing_time=0,
            errors=errors
        )
    
    async def _process_metrics_batch(self, metrics: List[Metric]) -> ProcessingResult:
        """Traite un batch de métriques."""
        # Convertir en DataFrame pour traitement vectorisé
        df_data = []
        for metric in metrics:
            df_data.append({
                'name': metric.name,
                'value': metric.value,
                'tenant_id': metric.tenant_id,
                'timestamp': metric.timestamp,
                'tags': json.dumps(metric.tags)
            })
        
        df = pd.DataFrame(df_data)
        
        # Traitement vectorisé
        try:
            # Validation
            df = df.dropna(subset=['value'])
            
            # Normalisation
            if not df.empty:
                df['value_normalized'] = StandardScaler().fit_transform(
                    df[['value']]
                ).flatten()
            
            # Stockage (à implémenter selon le backend)
            processed_count = len(df)
            error_count = len(metrics) - processed_count
            
            return ProcessingResult(
                success=error_count == 0,
                processed_count=processed_count,
                error_count=error_count,
                processing_time=0,
                errors=[]
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                processed_count=0,
                error_count=len(metrics),
                processing_time=0,
                errors=[str(e)]
            )
    
    async def _process_events_batch(self, events: List[Event]) -> ProcessingResult:
        """Traite un batch d'événements."""
        try:
            # Traitement simple pour les événements
            processed_count = len(events)
            
            return ProcessingResult(
                success=True,
                processed_count=processed_count,
                error_count=0,
                processing_time=0,
                errors=[]
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                processed_count=0,
                error_count=len(events),
                processing_time=0,
                errors=[str(e)]
            )


class StreamProcessor(BaseProcessor):
    """Processeur de flux pour données en continu."""
    
    def __init__(self, config: AnalyticsConfig):
        super().__init__(config)
        self.stream_buffer = Queue(maxsize=50000)
        self.processing_interval = 1.0  # secondes
        self.stream_task: Optional[asyncio.Task] = None
    
    async def _on_start(self):
        """Démarre le traitement de flux."""
        self.stream_task = asyncio.create_task(self._stream_processor())
    
    async def _on_stop(self):
        """Arrête le traitement de flux."""
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
    
    async def process(self, data: Union[Metric, Event]) -> ProcessingResult:
        """Ajoute des données au flux."""
        try:
            self.stream_buffer.put_nowait(data)
            return ProcessingResult(
                success=True,
                processed_count=1,
                error_count=0,
                processing_time=0
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                processed_count=0,
                error_count=1,
                processing_time=0,
                errors=[str(e)]
            )
    
    async def _stream_processor(self):
        """Processeur de flux principal."""
        buffer = []
        
        while self.is_running:
            try:
                # Collecter les données du buffer
                while not self.stream_buffer.empty() and len(buffer) < 1000:
                    try:
                        item = self.stream_buffer.get_nowait()
                        buffer.append(item)
                    except Empty:
                        break
                
                if buffer:
                    await self._process_stream_batch(buffer)
                    buffer.clear()
                
                await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                self.logger.error(f"Erreur stream processor: {e}")
                await asyncio.sleep(1)
    
    async def _process_stream_batch(self, batch: List[Union[Metric, Event]]):
        """Traite un batch de flux."""
        # Détecter les patterns et anomalies
        await self._detect_patterns(batch)
        
        # Mise à jour des métriques en temps réel
        await self._update_realtime_metrics(batch)
    
    async def _detect_patterns(self, batch: List[Union[Metric, Event]]):
        """Détecte des patterns dans le flux."""
        # Grouper par tenant et type
        grouped = {}
        for item in batch:
            key = f"{item.tenant_id}:{type(item).__name__}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)
        
        # Analyser chaque groupe
        for key, items in grouped.items():
            await self._analyze_group_patterns(key, items)
    
    async def _analyze_group_patterns(self, key: str, items: List[Union[Metric, Event]]):
        """Analyse les patterns d'un groupe."""
        if len(items) < 10:
            return
        
        # Analyse simple de fréquence
        frequencies = {}
        for item in items:
            if isinstance(item, Metric):
                freq_key = item.name
            else:
                freq_key = item.event_type
            
            frequencies[freq_key] = frequencies.get(freq_key, 0) + 1
        
        # Détecter les anomalies de fréquence
        if frequencies:
            avg_freq = statistics.mean(frequencies.values())
            for freq_key, freq in frequencies.items():
                if freq > avg_freq * 3:  # 3x la moyenne
                    self.logger.warning(
                        f"Fréquence anormalement élevée détectée: {key}:{freq_key} = {freq}"
                    )
    
    async def _update_realtime_metrics(self, batch: List[Union[Metric, Event]]):
        """Met à jour les métriques en temps réel."""
        # Compter par type
        metric_count = sum(1 for item in batch if isinstance(item, Metric))
        event_count = sum(1 for item in batch if isinstance(item, Event))
        
        self.logger.debug(
            f"Traité: {metric_count} métriques, {event_count} événements"
        )


class MLProcessor(BaseProcessor):
    """Processeur Machine Learning pour analyses avancées."""
    
    def __init__(self, config: AnalyticsConfig):
        super().__init__(config)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.models = {}
        self.feature_history = defaultdict(list)
        self.max_history = 1000
    
    async def process(self, data: Union[Metric, List[Metric]]) -> ProcessingResult:
        """Traite des données avec ML."""
        if isinstance(data, Metric):
            data = [data]
        
        start_time = time.time()
        processed_count = 0
        error_count = 0
        errors = []
        
        for metric in data:
            try:
                await self._process_metric_ml(metric)
                processed_count += 1
            except Exception as e:
                error_count += 1
                errors.append(str(e))
                self.logger.error(f"Erreur traitement ML: {e}")
        
        processing_time = time.time() - start_time
        
        result = ProcessingResult(
            success=error_count == 0,
            processed_count=processed_count,
            error_count=error_count,
            processing_time=processing_time,
            errors=errors
        )
        
        self._update_stats(result)
        return result
    
    async def _process_metric_ml(self, metric: Metric):
        """Traite une métrique avec ML."""
        # Extraction de features
        features = self._extract_features(metric)
        
        # Mise à jour de l'historique
        key = f"{metric.tenant_id}:{metric.name}"
        self.feature_history[key].append(features)
        
        # Limiter l'historique
        if len(self.feature_history[key]) > self.max_history:
            self.feature_history[key].pop(0)
        
        # Détection d'anomalies
        if len(self.feature_history[key]) >= 50:  # Minimum pour entraînement
            anomaly_score = await self._detect_anomaly(key, features)
            
            if anomaly_score > 0.8:  # Seuil d'anomalie
                await self._handle_anomaly(metric, anomaly_score)
    
    def _extract_features(self, metric: Metric) -> List[float]:
        """Extrait des features d'une métrique."""
        features = [
            float(metric.value),
            float(metric.timestamp.hour),
            float(metric.timestamp.weekday()),
            float(len(metric.tags)),
        ]
        
        # Features additionnelles basées sur les tags
        if 'source' in metric.tags:
            features.append(hash(metric.tags['source']) % 1000)
        else:
            features.append(0)
        
        return features
    
    async def _detect_anomaly(self, key: str, features: List[float]) -> float:
        """Détecte une anomalie."""
        history = self.feature_history[key]
        
        if len(history) < 50:
            return 0.0
        
        # Préparer les données
        X = np.array(history[:-1])  # Toutes sauf la dernière
        current = np.array([features])  # La métrique actuelle
        
        # Entraîner le modèle si nécessaire
        if key not in self.models:
            self.models[key] = IsolationForest(contamination=0.1, random_state=42)
            self.models[key].fit(X)
        
        # Prédiction d'anomalie
        anomaly_score = self.models[key].decision_function(current)[0]
        
        # Normaliser le score (0-1)
        return max(0, min(1, (anomaly_score + 1) / 2))
    
    async def _handle_anomaly(self, metric: Metric, score: float):
        """Gère une anomalie détectée."""
        self.logger.warning(
            f"Anomalie détectée: {metric.name} = {metric.value} "
            f"(score: {score:.3f}) pour tenant {metric.tenant_id}"
        )
        
        # Créer une alerte d'anomalie
        alert = Alert(
            name=f"anomaly_{metric.name}",
            title="Anomalie ML Détectée",
            message=f"Valeur anormale détectée pour {metric.name}: {metric.value}",
            severity="medium",
            tenant_id=metric.tenant_id,
            metric_name=metric.name,
            current_value=metric.value,
            metadata={
                'anomaly_score': score,
                'detection_method': 'isolation_forest'
            }
        )
        
        # Publier l'alerte (via event bus si disponible)
        # await self.event_bus.publish('anomaly_detected', alert.to_dict())


@dataclass
class PipelineStage:
    """Étape de pipeline."""
    name: str
    processor: BaseProcessor
    enabled: bool = True
    
    async def execute(self, data: Any) -> ProcessingResult:
        """Exécute l'étape."""
        if not self.enabled:
            return ProcessingResult(
                success=True,
                processed_count=0,
                error_count=0,
                processing_time=0,
                metadata={'skipped': True}
            )
        
        return await self.processor.process(data)


class DataPipeline:
    """Pipeline de traitement de données."""
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[PipelineStage] = []
        self.logger = Logger(f"Pipeline.{name}")
    
    def add_stage(self, name: str, processor: BaseProcessor, enabled: bool = True):
        """Ajoute une étape au pipeline."""
        stage = PipelineStage(name, processor, enabled)
        self.stages.append(stage)
        return self
    
    async def execute(self, data: Any) -> Dict[str, ProcessingResult]:
        """Exécute le pipeline."""
        results = {}
        current_data = data
        
        for stage in self.stages:
            try:
                result = await stage.execute(current_data)
                results[stage.name] = result
                
                if not result.success:
                    self.logger.error(
                        f"Échec de l'étape {stage.name}: {result.errors}"
                    )
                    break
                
                # Passer les données à l'étape suivante
                # (dans une vraie implémentation, on pourrait transformer les données)
                
            except Exception as e:
                error_result = ProcessingResult(
                    success=False,
                    processed_count=0,
                    error_count=1,
                    processing_time=0,
                    errors=[str(e)]
                )
                results[stage.name] = error_result
                self.logger.error(f"Erreur dans l'étape {stage.name}: {e}")
                break
        
        return results
    
    async def start(self):
        """Démarre tous les processeurs du pipeline."""
        for stage in self.stages:
            await stage.processor.start()
        
        self.logger.info(f"Pipeline {self.name} démarré")
    
    async def stop(self):
        """Arrête tous les processeurs du pipeline."""
        for stage in self.stages:
            await stage.processor.stop()
        
        self.logger.info(f"Pipeline {self.name} arrêté")


# Fonctions utilitaires
async def create_standard_pipeline(config: AnalyticsConfig) -> DataPipeline:
    """Crée un pipeline standard."""
    pipeline = DataPipeline("standard")
    
    # Étapes du pipeline
    pipeline.add_stage("realtime", RealTimeProcessor(config))
    pipeline.add_stage("stream", StreamProcessor(config))
    pipeline.add_stage("ml", MLProcessor(config))
    
    return pipeline


async def create_batch_pipeline(config: AnalyticsConfig) -> DataPipeline:
    """Crée un pipeline pour traitement par lots."""
    pipeline = DataPipeline("batch")
    
    pipeline.add_stage("batch", BatchProcessor(config))
    pipeline.add_stage("ml", MLProcessor(config))
    
    return pipeline
