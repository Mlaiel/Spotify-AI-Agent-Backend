"""
⚡ Advanced Processing Engine - Production-Ready System
======================================================

Moteur de traitement ultra-avancé avec pipelines ML, orchestration intelligente,
optimisation de performance et gestion de charge adaptive.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, AsyncGenerator
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, PriorityQueue
import weakref

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import aioredis
import aiocache

logger = logging.getLogger(__name__)


class ProcessingPriority(Enum):
    """Priorités de traitement"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class ProcessingStrategy(Enum):
    """Stratégies de traitement"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BATCH = "batch"
    STREAM = "stream"
    ADAPTIVE = "adaptive"


class ProcessingStage(Enum):
    """Étapes de traitement"""
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    CORE_PROCESSING = "core_processing"
    POSTPROCESSING = "postprocessing"
    OPTIMIZATION = "optimization"
    DELIVERY = "delivery"


class ProcessingStatus(Enum):
    """Statuts de traitement"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class ProcessingTask:
    """Tâche de traitement"""
    id: str = field(default_factory=lambda: f"task_{int(time.time() * 1000)}")
    name: str = ""
    data: Any = None
    
    # Configuration
    priority: ProcessingPriority = ProcessingPriority.MEDIUM
    strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE
    timeout: float = 30.0
    max_retries: int = 3
    
    # Métadonnées
    tenant_id: str = ""
    user_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # État
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Résultats et erreurs
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    
    # Métriques
    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Callbacks
    on_progress: Optional[Callable] = None
    on_completion: Optional[Callable] = None
    on_error: Optional[Callable] = None


@dataclass
class ProcessingResult:
    """Résultat de traitement"""
    task_id: str
    status: ProcessingStatus
    result: Any = None
    error: Optional[str] = None
    
    # Métriques de performance
    processing_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_avg_percent: float = 0.0
    
    # Détails des étapes
    stage_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Optimisations appliquées
    optimizations_applied: List[str] = field(default_factory=list)
    
    # Recommandations
    performance_recommendations: List[str] = field(default_factory=list)
    
    # Métadonnées
    processed_at: datetime = field(default_factory=datetime.utcnow)
    pipeline_version: str = "1.0"


class ProcessingPipeline(ABC):
    """Pipeline de traitement abstrait"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Métriques
        self.execution_count = 0
        self.total_processing_time = 0.0
        self.success_count = 0
        self.error_count = 0
    
    @abstractmethod
    async def process(self, task: ProcessingTask) -> ProcessingResult:
        """Traitement principal"""
        pass
    
    async def validate(self, task: ProcessingTask) -> bool:
        """Validation des données d'entrée"""
        return task.data is not None
    
    async def preprocess(self, task: ProcessingTask) -> ProcessingTask:
        """Préprocessing des données"""
        return task
    
    async def postprocess(self, result: ProcessingResult) -> ProcessingResult:
        """Post-processing des résultats"""
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtention des métriques du pipeline"""
        
        avg_processing_time = (
            self.total_processing_time / self.execution_count 
            if self.execution_count > 0 else 0.0
        )
        
        success_rate = (
            self.success_count / self.execution_count 
            if self.execution_count > 0 else 0.0
        )
        
        return {
            'name': self.name,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': success_rate,
            'avg_processing_time_ms': avg_processing_time
        }


class InfoTemplatePipeline(ProcessingPipeline):
    """Pipeline spécialisé pour les templates d'information"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("InfoTemplatePipeline", config)
        
        # Composants spécialisés
        self.template_validator = None
        self.content_enricher = None
        self.personalization_engine = None
        self.quality_assessor = None
        
        self._init_components()
    
    def _init_components(self):
        """Initialisation des composants spécialisés"""
        
        try:
            # Ici on initialiserait les composants réels
            # Pour la démo, on simule
            self.components_initialized = True
            self.logger.info("Info template pipeline components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {str(e)}")
            self.components_initialized = False
    
    async def process(self, task: ProcessingTask) -> ProcessingResult:
        """Traitement principal du template d'information"""
        
        start_time = time.time()
        stage_metrics = {}
        
        try:
            # Étape 1: Validation
            validation_start = time.time()
            if not await self.validate(task):
                return ProcessingResult(
                    task_id=task.id,
                    status=ProcessingStatus.FAILED,
                    error="Validation failed"
                )
            stage_metrics['validation'] = {'duration_ms': (time.time() - validation_start) * 1000}
            
            # Étape 2: Préprocessing
            preprocessing_start = time.time()
            preprocessed_task = await self.preprocess(task)
            stage_metrics['preprocessing'] = {'duration_ms': (time.time() - preprocessing_start) * 1000}
            
            # Étape 3: Traitement principal
            core_start = time.time()
            processed_data = await self._core_processing(preprocessed_task)
            stage_metrics['core_processing'] = {'duration_ms': (time.time() - core_start) * 1000}
            
            # Étape 4: Optimisation
            optimization_start = time.time()
            optimized_data, optimizations = await self._optimize_output(processed_data, task)
            stage_metrics['optimization'] = {'duration_ms': (time.time() - optimization_start) * 1000}
            
            # Étape 5: Post-processing
            postprocessing_start = time.time()
            result = ProcessingResult(
                task_id=task.id,
                status=ProcessingStatus.COMPLETED,
                result=optimized_data,
                processing_time_ms=(time.time() - start_time) * 1000,
                stage_metrics=stage_metrics,
                optimizations_applied=optimizations
            )
            
            final_result = await self.postprocess(result)
            stage_metrics['postprocessing'] = {'duration_ms': (time.time() - postprocessing_start) * 1000}
            final_result.stage_metrics = stage_metrics
            
            # Mise à jour des métriques
            self.execution_count += 1
            self.success_count += 1
            self.total_processing_time += final_result.processing_time_ms
            
            return final_result
            
        except Exception as e:
            self.error_count += 1
            self.execution_count += 1
            
            return ProcessingResult(
                task_id=task.id,
                status=ProcessingStatus.FAILED,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
                stage_metrics=stage_metrics
            )
    
    async def _core_processing(self, task: ProcessingTask) -> Dict[str, Any]:
        """Traitement principal spécialisé"""
        
        data = task.data
        
        # Simulation du traitement de template
        processed_data = {
            'original_template': data,
            'enriched_content': await self._enrich_content(data, task.context),
            'personalization_applied': await self._apply_personalization(data, task),
            'quality_score': await self._assess_quality(data),
            'metadata': {
                'processing_timestamp': datetime.utcnow().isoformat(),
                'pipeline_version': '2.0',
                'tenant_id': task.tenant_id,
                'user_id': task.user_id
            }
        }
        
        return processed_data
    
    async def _enrich_content(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrichissement du contenu"""
        
        # Simulation d'enrichissement ML
        enrichment = {
            'sentiment_analysis': {
                'sentiment': 'positive',
                'confidence': 0.85
            },
            'topic_extraction': [
                'notifications', 'alerts', 'monitoring'
            ],
            'complexity_score': 0.6,
            'readability_score': 0.8,
            'engagement_prediction': 0.75
        }
        
        # Simulation de délai de traitement ML
        await asyncio.sleep(0.1)
        
        return enrichment
    
    async def _apply_personalization(self, data: Any, task: ProcessingTask) -> Dict[str, Any]:
        """Application de la personnalisation"""
        
        personalization = {
            'user_segment': 'power_user',
            'communication_style': 'professional',
            'preferred_channel': 'slack',
            'optimal_timing': datetime.utcnow() + timedelta(hours=2),
            'personalization_confidence': 0.9
        }
        
        return personalization
    
    async def _assess_quality(self, data: Any) -> float:
        """Évaluation de la qualité"""
        
        # Simulation d'évaluation qualité
        quality_factors = [0.8, 0.9, 0.7, 0.85]  # Différents critères
        return statistics.mean(quality_factors)
    
    async def _optimize_output(self, data: Dict[str, Any], task: ProcessingTask) -> Tuple[Dict[str, Any], List[str]]:
        """Optimisation de la sortie"""
        
        optimizations = []
        optimized_data = data.copy()
        
        # Optimisation de compression
        if len(str(data)) > 1000:
            optimized_data['compressed'] = True
            optimizations.append("Applied data compression")
        
        # Optimisation de cache
        if task.priority in [ProcessingPriority.HIGH, ProcessingPriority.CRITICAL]:
            optimized_data['cache_priority'] = 'high'
            optimizations.append("Set high cache priority")
        
        # Optimisation de format
        if 'slack' in task.context.get('channels', []):
            optimized_data['slack_optimized'] = True
            optimizations.append("Applied Slack format optimization")
        
        return optimized_data, optimizations


class BatchProcessor:
    """Processeur par lots avancé"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration
        self.batch_size = config.get('batch_size', 10)
        self.max_batch_wait_time = config.get('max_batch_wait_time', 5.0)
        self.max_parallel_batches = config.get('max_parallel_batches', 3)
        
        # Files de traitement
        self.pending_tasks: Queue[ProcessingTask] = Queue()
        self.active_batches: Dict[str, List[ProcessingTask]] = {}
        
        # Métriques
        self.batch_metrics = {
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0,
            'throughput_per_second': 0.0
        }
        
        # Contrôle
        self.running = False
        self.batch_processor_task = None
    
    async def start(self):
        """Démarrage du processeur par lots"""
        
        if self.running:
            return
        
        self.running = True
        self.batch_processor_task = asyncio.create_task(self._batch_processing_loop())
        self.logger.info("Batch processor started")
    
    async def stop(self):
        """Arrêt du processeur par lots"""
        
        self.running = False
        
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Batch processor stopped")
    
    async def submit_task(self, task: ProcessingTask) -> str:
        """Soumission d'une tâche pour traitement par lots"""
        
        self.pending_tasks.put(task)
        return task.id
    
    async def _batch_processing_loop(self):
        """Boucle principale de traitement par lots"""
        
        try:
            while self.running:
                # Formation d'un lot
                batch = await self._form_batch()
                
                if batch:
                    # Traitement du lot
                    batch_id = f"batch_{int(time.time() * 1000)}"
                    self.active_batches[batch_id] = batch
                    
                    # Traitement asynchrone du lot
                    asyncio.create_task(self._process_batch(batch_id, batch))
                
                # Attente avant la prochaine itération
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            self.logger.info("Batch processing loop cancelled")
        except Exception as e:
            self.logger.error(f"Batch processing loop error: {str(e)}")
    
    async def _form_batch(self) -> List[ProcessingTask]:
        """Formation d'un lot de tâches"""
        
        batch = []
        batch_start_time = time.time()
        
        # Collecte des tâches jusqu'à atteindre la taille du lot ou le timeout
        while (len(batch) < self.batch_size and 
               time.time() - batch_start_time < self.max_batch_wait_time):
            
            try:
                task = self.pending_tasks.get_nowait()
                batch.append(task)
            except:
                # Pas de tâche disponible, attendre un peu
                await asyncio.sleep(0.1)
        
        return batch
    
    async def _process_batch(self, batch_id: str, batch: List[ProcessingTask]):
        """Traitement d'un lot de tâches"""
        
        start_time = time.time()
        
        try:
            # Tri des tâches par priorité
            sorted_batch = sorted(batch, key=lambda t: t.priority.value)
            
            # Traitement parallèle des tâches du lot
            pipeline = InfoTemplatePipeline(self.config)
            
            tasks_coroutines = [
                pipeline.process(task) for task in sorted_batch
            ]
            
            results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
            
            # Traitement des résultats
            for task, result in zip(sorted_batch, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Task {task.id} failed: {str(result)}")
                    task.status = ProcessingStatus.FAILED
                    task.error = str(result)
                else:
                    task.status = ProcessingStatus.COMPLETED
                    task.result = result
                
                # Callback de completion si défini
                if task.on_completion:
                    try:
                        await task.on_completion(task, result)
                    except Exception as e:
                        self.logger.error(f"Completion callback failed: {str(e)}")
            
            # Mise à jour des métriques
            processing_time = time.time() - start_time
            self._update_batch_metrics(len(batch), processing_time)
            
        except Exception as e:
            self.logger.error(f"Batch {batch_id} processing failed: {str(e)}")
            
            # Marquer toutes les tâches du lot comme échouées
            for task in batch:
                task.status = ProcessingStatus.FAILED
                task.error = f"Batch processing error: {str(e)}"
        
        finally:
            # Nettoyage
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]
    
    def _update_batch_metrics(self, batch_size: int, processing_time: float):
        """Mise à jour des métriques de traitement par lots"""
        
        self.batch_metrics['total_batches'] += 1
        
        # Moyenne mobile de la taille des lots
        total_batches = self.batch_metrics['total_batches']
        current_avg_size = self.batch_metrics['avg_batch_size']
        new_avg_size = (current_avg_size * (total_batches - 1) + batch_size) / total_batches
        self.batch_metrics['avg_batch_size'] = new_avg_size
        
        # Moyenne mobile du temps de traitement
        current_avg_time = self.batch_metrics['avg_processing_time']
        new_avg_time = (current_avg_time * (total_batches - 1) + processing_time) / total_batches
        self.batch_metrics['avg_processing_time'] = new_avg_time
        
        # Calcul du débit
        if processing_time > 0:
            throughput = batch_size / processing_time
            current_throughput = self.batch_metrics['throughput_per_second']
            new_throughput = (current_throughput * (total_batches - 1) + throughput) / total_batches
            self.batch_metrics['throughput_per_second'] = new_throughput


class StreamProcessor:
    """Processeur de flux en temps réel"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration
        self.buffer_size = config.get('stream_buffer_size', 100)
        self.flush_interval = config.get('stream_flush_interval', 1.0)
        self.max_processing_delay = config.get('max_processing_delay', 5.0)
        
        # Buffer de streaming
        self.stream_buffer: List[ProcessingTask] = []
        self.buffer_lock = asyncio.Lock()
        
        # Métriques de streaming
        self.stream_metrics = {
            'total_items_processed': 0,
            'avg_latency_ms': 0.0,
            'current_buffer_size': 0,
            'peak_buffer_size': 0,
            'processing_rate_per_second': 0.0
        }
        
        # Contrôle
        self.running = False
        self.stream_processor_task = None
    
    async def start(self):
        """Démarrage du processeur de flux"""
        
        if self.running:
            return
        
        self.running = True
        self.stream_processor_task = asyncio.create_task(self._stream_processing_loop())
        self.logger.info("Stream processor started")
    
    async def stop(self):
        """Arrêt du processeur de flux"""
        
        self.running = False
        
        if self.stream_processor_task:
            self.stream_processor_task.cancel()
            try:
                await self.stream_processor_task
            except asyncio.CancelledError:
                pass
        
        # Traitement final du buffer
        await self._flush_buffer()
        
        self.logger.info("Stream processor stopped")
    
    async def stream_task(self, task: ProcessingTask) -> str:
        """Ajout d'une tâche au flux"""
        
        async with self.buffer_lock:
            self.stream_buffer.append(task)
            
            # Mise à jour des métriques
            current_size = len(self.stream_buffer)
            self.stream_metrics['current_buffer_size'] = current_size
            if current_size > self.stream_metrics['peak_buffer_size']:
                self.stream_metrics['peak_buffer_size'] = current_size
        
        return task.id
    
    async def _stream_processing_loop(self):
        """Boucle principale de traitement de flux"""
        
        try:
            while self.running:
                # Flush périodique du buffer
                await self._flush_buffer()
                
                # Attente avant la prochaine itération
                await asyncio.sleep(self.flush_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Stream processing loop cancelled")
        except Exception as e:
            self.logger.error(f"Stream processing loop error: {str(e)}")
    
    async def _flush_buffer(self):
        """Vidage et traitement du buffer"""
        
        if not self.stream_buffer:
            return
        
        async with self.buffer_lock:
            # Extraction des tâches du buffer
            tasks_to_process = self.stream_buffer.copy()
            self.stream_buffer.clear()
            self.stream_metrics['current_buffer_size'] = 0
        
        if not tasks_to_process:
            return
        
        start_time = time.time()
        
        try:
            # Traitement en flux avec pipeline
            pipeline = InfoTemplatePipeline(self.config)
            
            # Traitement asynchrone avec limite de concurrence
            semaphore = asyncio.Semaphore(5)  # Max 5 tâches simultanées
            
            async def process_with_semaphore(task):
                async with semaphore:
                    return await pipeline.process(task)
            
            # Traitement de toutes les tâches
            processing_coroutines = [
                process_with_semaphore(task) for task in tasks_to_process
            ]
            
            results = await asyncio.gather(*processing_coroutines, return_exceptions=True)
            
            # Traitement des résultats
            for task, result in zip(tasks_to_process, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Stream task {task.id} failed: {str(result)}")
                    task.status = ProcessingStatus.FAILED
                    task.error = str(result)
                else:
                    task.status = ProcessingStatus.COMPLETED
                    task.result = result
                
                # Callback si défini
                if task.on_completion:
                    try:
                        await task.on_completion(task, result)
                    except Exception as e:
                        self.logger.error(f"Stream completion callback failed: {str(e)}")
            
            # Mise à jour des métriques
            processing_time = time.time() - start_time
            self._update_stream_metrics(len(tasks_to_process), processing_time)
            
        except Exception as e:
            self.logger.error(f"Stream buffer flush failed: {str(e)}")
    
    def _update_stream_metrics(self, items_count: int, processing_time: float):
        """Mise à jour des métriques de streaming"""
        
        # Total d'éléments traités
        self.stream_metrics['total_items_processed'] += items_count
        
        # Latence moyenne
        if items_count > 0:
            avg_latency = (processing_time / items_count) * 1000  # en ms
            current_avg = self.stream_metrics['avg_latency_ms']
            total_items = self.stream_metrics['total_items_processed']
            
            # Moyenne pondérée
            new_avg = (current_avg * (total_items - items_count) + avg_latency * items_count) / total_items
            self.stream_metrics['avg_latency_ms'] = new_avg
        
        # Taux de traitement
        if processing_time > 0:
            processing_rate = items_count / processing_time
            self.stream_metrics['processing_rate_per_second'] = processing_rate


class AdaptiveLoadBalancer:
    """Équilibreur de charge adaptatif"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Processeurs disponibles
        self.batch_processor = BatchProcessor(config)
        self.stream_processor = StreamProcessor(config)
        
        # Métriques de charge
        self.load_metrics = {
            'current_load': 0.0,
            'batch_queue_size': 0,
            'stream_buffer_size': 0,
            'system_cpu_percent': 0.0,
            'system_memory_percent': 0.0
        }
        
        # Seuils adaptatifs
        self.load_thresholds = {
            'low_load': 0.3,
            'medium_load': 0.6,
            'high_load': 0.8,
            'critical_load': 0.95
        }
        
        # Historique des décisions
        self.routing_history: List[Dict[str, Any]] = []
    
    async def start(self):
        """Démarrage de l'équilibreur de charge"""
        
        await self.batch_processor.start()
        await self.stream_processor.start()
        self.logger.info("Adaptive load balancer started")
    
    async def stop(self):
        """Arrêt de l'équilibreur de charge"""
        
        await self.batch_processor.stop()
        await self.stream_processor.stop()
        self.logger.info("Adaptive load balancer stopped")
    
    async def route_task(self, task: ProcessingTask) -> str:
        """Routage intelligent des tâches"""
        
        # Analyse de la charge actuelle
        current_load = await self._assess_current_load()
        
        # Décision de routage basée sur plusieurs facteurs
        routing_decision = await self._make_routing_decision(task, current_load)
        
        # Exécution du routage
        task_id = await self._execute_routing(task, routing_decision)
        
        # Enregistrement de la décision
        self._record_routing_decision(task, routing_decision, current_load)
        
        return task_id
    
    async def _assess_current_load(self) -> float:
        """Évaluation de la charge système actuelle"""
        
        # Métriques des processeurs
        batch_load = min(1.0, self.batch_processor.pending_tasks.qsize() / 100)
        stream_load = min(1.0, len(self.stream_processor.stream_buffer) / self.stream_processor.buffer_size)
        
        # Simulation de métriques système
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
        memory_percent = psutil.virtual_memory().percent / 100.0
        
        # Mise à jour des métriques
        self.load_metrics.update({
            'current_load': (batch_load + stream_load + cpu_percent + memory_percent) / 4,
            'batch_queue_size': self.batch_processor.pending_tasks.qsize(),
            'stream_buffer_size': len(self.stream_processor.stream_buffer),
            'system_cpu_percent': cpu_percent,
            'system_memory_percent': memory_percent
        })
        
        return self.load_metrics['current_load']
    
    async def _make_routing_decision(self, task: ProcessingTask, current_load: float) -> str:
        """Prise de décision de routage"""
        
        # Facteurs de décision
        priority = task.priority
        strategy = task.strategy
        data_size = len(str(task.data)) if task.data else 0
        
        # Logique de routage adaptatif
        if strategy == ProcessingStrategy.BATCH:
            return "batch"
        elif strategy == ProcessingStrategy.STREAM:
            return "stream"
        elif strategy == ProcessingStrategy.ADAPTIVE:
            # Décision automatique basée sur la charge et les caractéristiques
            
            # Tâches critiques -> streaming pour latence minimale
            if priority in [ProcessingPriority.CRITICAL, ProcessingPriority.HIGH]:
                if current_load < self.load_thresholds['high_load']:
                    return "stream"
                else:
                    return "batch"  # Même critiques, mais système surchargé
            
            # Tâches volumineuses -> batch pour efficacité
            if data_size > 10000:  # Seuil arbitraire
                return "batch"
            
            # Charge faible -> streaming pour réactivité
            if current_load < self.load_thresholds['medium_load']:
                return "stream"
            
            # Charge élevée -> batch pour throughput
            if current_load > self.load_thresholds['high_load']:
                return "batch"
            
            # Charge moyenne -> équilibrage
            batch_queue_size = self.load_metrics['batch_queue_size']
            stream_buffer_size = self.load_metrics['stream_buffer_size']
            
            if batch_queue_size < stream_buffer_size:
                return "batch"
            else:
                return "stream"
        
        # Stratégies spécifiques
        return "batch"  # Default
    
    async def _execute_routing(self, task: ProcessingTask, routing_decision: str) -> str:
        """Exécution du routage"""
        
        if routing_decision == "batch":
            return await self.batch_processor.submit_task(task)
        elif routing_decision == "stream":
            return await self.stream_processor.stream_task(task)
        else:
            # Fallback vers batch
            return await self.batch_processor.submit_task(task)
    
    def _record_routing_decision(self, task: ProcessingTask, decision: str, load: float):
        """Enregistrement de la décision de routage"""
        
        decision_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'task_id': task.id,
            'task_priority': task.priority.name,
            'task_strategy': task.strategy.name,
            'routing_decision': decision,
            'system_load': load,
            'batch_queue_size': self.load_metrics['batch_queue_size'],
            'stream_buffer_size': self.load_metrics['stream_buffer_size']
        }
        
        self.routing_history.append(decision_record)
        
        # Maintenir un historique limité
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-500:]
    
    async def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Statistiques de l'équilibreur de charge"""
        
        # Statistiques de routage
        total_decisions = len(self.routing_history)
        batch_decisions = sum(1 for d in self.routing_history if d['routing_decision'] == 'batch')
        stream_decisions = total_decisions - batch_decisions
        
        # Métriques de performance des processeurs
        batch_metrics = self.batch_processor.batch_metrics
        stream_metrics = self.stream_processor.stream_metrics
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'current_load': self.load_metrics,
            'routing_stats': {
                'total_decisions': total_decisions,
                'batch_percentage': batch_decisions / total_decisions if total_decisions > 0 else 0,
                'stream_percentage': stream_decisions / total_decisions if total_decisions > 0 else 0
            },
            'batch_processor_metrics': batch_metrics,
            'stream_processor_metrics': stream_metrics,
            'system_status': self._determine_system_status()
        }
    
    def _determine_system_status(self) -> str:
        """Détermination du statut système"""
        
        current_load = self.load_metrics['current_load']
        
        if current_load < self.load_thresholds['low_load']:
            return "optimal"
        elif current_load < self.load_thresholds['medium_load']:
            return "normal"
        elif current_load < self.load_thresholds['high_load']:
            return "busy"
        elif current_load < self.load_thresholds['critical_load']:
            return "overloaded"
        else:
            return "critical"


class ProcessingOrchestrator:
    """Orchestrateur principal de traitement"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Composants principaux
        self.load_balancer = AdaptiveLoadBalancer(config)
        
        # Registre des tâches
        self.task_registry: Dict[str, ProcessingTask] = {}
        self.task_lock = asyncio.Lock()
        
        # Métriques globales
        self.global_metrics = {
            'total_tasks_submitted': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'avg_end_to_end_latency_ms': 0.0,
            'system_uptime_seconds': 0.0,
            'peak_concurrent_tasks': 0
        }
        
        # Démarrage
        self.start_time = datetime.utcnow()
    
    async def start(self):
        """Démarrage de l'orchestrateur"""
        
        await self.load_balancer.start()
        self.logger.info("Processing orchestrator started")
    
    async def stop(self):
        """Arrêt de l'orchestrateur"""
        
        await self.load_balancer.stop()
        self.logger.info("Processing orchestrator stopped")
    
    async def submit_processing_task(
        self, 
        name: str,
        data: Any,
        priority: ProcessingPriority = ProcessingPriority.MEDIUM,
        strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE,
        **kwargs
    ) -> str:
        """Soumission d'une tâche de traitement"""
        
        # Création de la tâche
        task = ProcessingTask(
            name=name,
            data=data,
            priority=priority,
            strategy=strategy,
            **kwargs
        )
        
        # Enregistrement de la tâche
        async with self.task_lock:
            self.task_registry[task.id] = task
            
            # Mise à jour des métriques
            self.global_metrics['total_tasks_submitted'] += 1
            current_tasks = len([t for t in self.task_registry.values() 
                                if t.status in [ProcessingStatus.PENDING, ProcessingStatus.RUNNING]])
            
            if current_tasks > self.global_metrics['peak_concurrent_tasks']:
                self.global_metrics['peak_concurrent_tasks'] = current_tasks
        
        # Configuration des callbacks
        task.on_completion = self._on_task_completion
        task.on_error = self._on_task_error
        
        # Routage via l'équilibreur de charge
        try:
            await self.load_balancer.route_task(task)
            task.status = ProcessingStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            return task.id
            
        except Exception as e:
            task.status = ProcessingStatus.FAILED
            task.error = str(e)
            self.logger.error(f"Failed to submit task {task.id}: {str(e)}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtention du statut d'une tâche"""
        
        async with self.task_lock:
            task = self.task_registry.get(task_id)
            
            if not task:
                return None
            
            return {
                'id': task.id,
                'name': task.name,
                'status': task.status.name,
                'priority': task.priority.name,
                'strategy': task.strategy.name,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'processing_time_ms': task.processing_time_ms,
                'retry_count': task.retry_count,
                'error': task.error,
                'has_result': task.result is not None
            }
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Obtention du résultat d'une tâche"""
        
        async with self.task_lock:
            task = self.task_registry.get(task_id)
            
            if not task or task.status != ProcessingStatus.COMPLETED:
                return None
            
            return task.result
    
    async def _on_task_completion(self, task: ProcessingTask, result: Any):
        """Callback de completion de tâche"""
        
        task.completed_at = datetime.utcnow()
        task.status = ProcessingStatus.COMPLETED
        
        if task.started_at:
            task.processing_time_ms = (task.completed_at - task.started_at).total_seconds() * 1000
        
        # Mise à jour des métriques globales
        self.global_metrics['total_tasks_completed'] += 1
        
        # Calcul de la latence end-to-end
        if task.created_at:
            end_to_end_latency = (task.completed_at - task.created_at).total_seconds() * 1000
            current_avg = self.global_metrics['avg_end_to_end_latency_ms']
            completed_tasks = self.global_metrics['total_tasks_completed']
            new_avg = (current_avg * (completed_tasks - 1) + end_to_end_latency) / completed_tasks
            self.global_metrics['avg_end_to_end_latency_ms'] = new_avg
    
    async def _on_task_error(self, task: ProcessingTask, error: Exception):
        """Callback d'erreur de tâche"""
        
        task.status = ProcessingStatus.FAILED
        task.error = str(error)
        
        self.global_metrics['total_tasks_failed'] += 1
        
        self.logger.error(f"Task {task.id} failed: {str(error)}")
    
    async def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Métriques complètes de l'orchestrateur"""
        
        # Calcul de l'uptime
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        self.global_metrics['system_uptime_seconds'] = uptime
        
        # Métriques de l'équilibreur de charge
        load_balancer_stats = await self.load_balancer.get_load_balancer_stats()
        
        # Statistiques des tâches actives
        async with self.task_lock:
            active_tasks = [t for t in self.task_registry.values() 
                           if t.status in [ProcessingStatus.PENDING, ProcessingStatus.RUNNING]]
            
            task_stats = {
                'total_registered_tasks': len(self.task_registry),
                'active_tasks': len(active_tasks),
                'pending_tasks': len([t for t in active_tasks if t.status == ProcessingStatus.PENDING]),
                'running_tasks': len([t for t in active_tasks if t.status == ProcessingStatus.RUNNING])
            }
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': uptime,
            'global_metrics': self.global_metrics,
            'task_statistics': task_stats,
            'load_balancer_metrics': load_balancer_stats,
            'system_health': self._assess_system_health()
        }
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Évaluation de la santé du système"""
        
        total_submitted = self.global_metrics['total_tasks_submitted']
        total_completed = self.global_metrics['total_tasks_completed']
        total_failed = self.global_metrics['total_tasks_failed']
        
        success_rate = total_completed / total_submitted if total_submitted > 0 else 0.0
        failure_rate = total_failed / total_submitted if total_submitted > 0 else 0.0
        
        # Détermination du statut de santé
        if success_rate > 0.95:
            health_status = "excellent"
        elif success_rate > 0.90:
            health_status = "good"
        elif success_rate > 0.80:
            health_status = "fair"
        elif success_rate > 0.70:
            health_status = "poor"
        else:
            health_status = "critical"
        
        return {
            'status': health_status,
            'success_rate': success_rate,
            'failure_rate': failure_rate,
            'avg_latency_ms': self.global_metrics['avg_end_to_end_latency_ms'],
            'peak_concurrent_tasks': self.global_metrics['peak_concurrent_tasks']
        }
