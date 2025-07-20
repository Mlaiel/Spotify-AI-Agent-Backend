"""
Processeur de flux temps réel pour alertes - Spotify AI Agent
Traitement streaming haute performance avec Apache Kafka et analytics en temps réel
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import aioredis
import asyncpg
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
from prometheus_client import Counter, Histogram, Gauge
import numpy as np
from collections import deque, defaultdict
import hashlib

from ..models.alert_models import AlertEvent, AlertSeverity, AlertStatus
from ..anomaly_detector import AnomalyDetector
from ..correlation_analyzer import CorrelationAnalyzer

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Modes de traitement"""
    REAL_TIME = "real_time"
    MICRO_BATCH = "micro_batch"
    SLIDING_WINDOW = "sliding_window"

class WindowType(Enum):
    """Types de fenêtres temporelles"""
    TUMBLING = "tumbling"  # Fenêtres non-chevauchantes
    SLIDING = "sliding"    # Fenêtres chevauchantes
    SESSION = "session"    # Basées sur l'activité

@dataclass
class StreamingWindow:
    """Fenêtre de streaming pour agrégation"""
    window_id: str
    start_time: datetime
    end_time: datetime
    events: List[AlertEvent] = field(default_factory=list)
    aggregated_metrics: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False

@dataclass
class ProcessingMetrics:
    """Métriques de traitement"""
    events_processed: int = 0
    events_per_second: float = 0.0
    processing_latency_ms: float = 0.0
    error_count: int = 0
    last_processing_time: Optional[datetime] = None

class AlertStreamProcessor:
    """
    Processeur de flux temps réel pour alertes
    
    Fonctionnalités:
    - Traitement streaming haute performance
    - Détection d'anomalies en temps réel
    - Agrégation par fenêtres temporelles
    - Corrélation d'événements en streaming
    - Back-pressure handling et scaling automatique
    - Pattern detection et alerting précoce
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processing_mode = ProcessingMode(config.get('processing_mode', 'real_time'))
        self.window_size = timedelta(seconds=config.get('window_size_seconds', 60))
        self.slide_interval = timedelta(seconds=config.get('slide_interval_seconds', 10))
        
        # Composants
        self.redis_client: Optional[aioredis.Redis] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.correlation_analyzer: Optional[CorrelationAnalyzer] = None
        
        # Kafka
        self.kafka_producer: Optional[KafkaProducer] = None
        self.kafka_consumer: Optional[KafkaConsumer] = None
        
        # État du streaming
        self.active_windows: Dict[str, StreamingWindow] = {}
        self.event_buffer: deque = deque(maxlen=config.get('buffer_size', 10000))
        self.processing_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Métriques et monitoring
        self.processing_metrics = ProcessingMetrics()
        self.rate_limiter = deque(maxlen=1000)  # Pour calcul de débit
        
        # Handlers configurables
        self.event_handlers: List[Callable] = []
        self.anomaly_handlers: List[Callable] = []
        self.correlation_handlers: List[Callable] = []
        
        # Métriques Prometheus
        self.events_processed_total = Counter(
            'stream_events_processed_total',
            'Total des événements traités en streaming',
            ['severity', 'source', 'processing_result']
        )
        self.processing_latency = Histogram(
            'stream_processing_latency_seconds',
            'Latence de traitement streaming'
        )
        self.active_windows_gauge = Gauge(
            'stream_active_windows',
            'Nombre de fenêtres actives'
        )
        self.throughput_gauge = Gauge(
            'stream_throughput_events_per_second',
            'Débit de traitement (événements/seconde)'
        )
        
    async def initialize(self):
        """Initialisation du processeur de streaming"""
        try:
            # Connexions
            await self._initialize_connections()
            
            # Composants ML
            await self._initialize_ml_components()
            
            # Kafka
            await self._initialize_kafka()
            
            # Fenêtres initiales
            await self._initialize_windows()
            
            logger.info("Processeur de streaming initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation streaming: {e}")
            raise
    
    async def _initialize_connections(self):
        """Initialisation des connexions"""
        # Redis pour cache et état
        self.redis_client = await aioredis.from_url(
            self.config.get('redis_url', 'redis://localhost:6379'),
            encoding='utf-8',
            decode_responses=True
        )
        
        # PostgreSQL pour persistance
        self.db_pool = await asyncpg.create_pool(
            self.config.get('database_url'),
            min_size=5,
            max_size=20
        )
    
    async def _initialize_ml_components(self):
        """Initialisation des composants ML"""
        # Détecteur d'anomalies
        if self.config.get('enable_anomaly_detection', True):
            from ..anomaly_detector import create_anomaly_detector
            self.anomaly_detector = await create_anomaly_detector(
                self.config.get('anomaly_detection', {})
            )
        
        # Analyseur de corrélation
        if self.config.get('enable_correlation_analysis', True):
            self.correlation_analyzer = CorrelationAnalyzer(
                self.config.get('correlation_analysis', {})
            )
            await self.correlation_analyzer.initialize()
    
    async def _initialize_kafka(self):
        """Initialisation Kafka"""
        try:
            kafka_config = self.config.get('kafka', {})
            
            # Producer pour output
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_config.get('bootstrap_servers', ['localhost:9092']),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                batch_size=16384,
                linger_ms=10
            )
            
            logger.info("Kafka producer initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation Kafka: {e}")
    
    async def _initialize_windows(self):
        """Initialisation des fenêtres de streaming"""
        if self.processing_mode in [ProcessingMode.MICRO_BATCH, ProcessingMode.SLIDING_WINDOW]:
            current_time = datetime.now()
            window_id = self._generate_window_id(current_time)
            
            window = StreamingWindow(
                window_id=window_id,
                start_time=current_time,
                end_time=current_time + self.window_size
            )
            
            self.active_windows[window_id] = window
    
    async def start_processing(self):
        """Démarrage du traitement streaming"""
        if self.is_running:
            logger.warning("Le processeur est déjà en cours d'exécution")
            return
        
        try:
            self.is_running = True
            
            # Démarrage des tâches parallèles
            tasks = [
                asyncio.create_task(self._event_processing_loop()),
                asyncio.create_task(self._window_management_loop()),
                asyncio.create_task(self._metrics_update_loop()),
                asyncio.create_task(self._health_check_loop())
            ]
            
            if self.config.get('enable_kafka_consumer', False):
                tasks.append(asyncio.create_task(self._kafka_consumer_loop()))
            
            self.processing_tasks = tasks
            
            logger.info("Traitement streaming démarré")
            
            # Attendre que toutes les tâches se terminent
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Erreur durant le traitement streaming: {e}")
            raise
        finally:
            self.is_running = False
    
    async def stop_processing(self):
        """Arrêt du traitement streaming"""
        logger.info("Arrêt du traitement streaming...")
        
        self.is_running = False
        
        # Annulation des tâches
        for task in self.processing_tasks:
            if not task.done():
                task.cancel()
        
        # Attendre l'arrêt complet
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Finalisation des fenêtres actives
        await self._finalize_active_windows()
        
        # Fermeture des connexions
        await self._cleanup_connections()
        
        logger.info("Traitement streaming arrêté")
    
    async def process_event(self, alert_event: AlertEvent) -> Dict[str, Any]:
        """
        Traitement d'un événement d'alerte unique
        
        Args:
            alert_event: Événement à traiter
            
        Returns:
            Dict contenant les résultats du traitement
        """
        start_time = datetime.now()
        
        try:
            # Ajout à la file de traitement
            self.event_buffer.append(alert_event)
            self.rate_limiter.append(start_time)
            
            # Traitement selon le mode
            if self.processing_mode == ProcessingMode.REAL_TIME:
                result = await self._process_real_time(alert_event)
            else:
                result = await self._process_windowed(alert_event)
            
            # Mise à jour des métriques
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_latency.observe(processing_time)
            self.events_processed_total.labels(
                severity=alert_event.severity.value,
                source=alert_event.source.value,
                processing_result='success'
            ).inc()
            
            # Mise à jour des métriques internes
            self.processing_metrics.events_processed += 1
            self.processing_metrics.processing_latency_ms = processing_time * 1000
            self.processing_metrics.last_processing_time = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur traitement événement {alert_event.id}: {e}")
            self.processing_metrics.error_count += 1
            self.events_processed_total.labels(
                severity=alert_event.severity.value,
                source=alert_event.source.value,
                processing_result='error'
            ).inc()
            
            return {'error': str(e), 'event_id': alert_event.id}
    
    async def _process_real_time(self, alert_event: AlertEvent) -> Dict[str, Any]:
        """Traitement temps réel"""
        result = {
            'event_id': alert_event.id,
            'processing_mode': 'real_time',
            'processed_at': datetime.now().isoformat(),
            'analyses': {}
        }
        
        # Détection d'anomalie
        if self.anomaly_detector:
            try:
                anomaly_result = await self.anomaly_detector.detect_anomaly(
                    self._convert_to_time_series_point(alert_event)
                )
                result['analyses']['anomaly'] = {
                    'score': anomaly_result.anomaly_score,
                    'severity': anomaly_result.severity_level,
                    'confidence': anomaly_result.confidence
                }
                
                # Notification si anomalie significative
                if anomaly_result.anomaly_score > 0.8:
                    await self._notify_anomaly(alert_event, anomaly_result)
                    
            except Exception as e:
                logger.error(f"Erreur détection anomalie: {e}")
        
        # Analyse de corrélation
        if self.correlation_analyzer:
            try:
                correlation_result = await self.correlation_analyzer.analyze_correlations(
                    self._convert_to_correlation_event(alert_event)
                )
                result['analyses']['correlation'] = {
                    'correlated_events_count': len(correlation_result.correlated_events),
                    'correlation_strength': max(correlation_result.correlation_scores.values()) if correlation_result.correlation_scores else 0.0
                }
                
                # Notification si corrélation forte
                if correlation_result.correlated_events:
                    await self._notify_correlation(alert_event, correlation_result)
                    
            except Exception as e:
                logger.error(f"Erreur analyse corrélation: {e}")
        
        # Enrichissement contextuel
        await self._enrich_event_context(alert_event, result)
        
        # Publication des résultats
        await self._publish_results(alert_event, result)
        
        return result
    
    async def _process_windowed(self, alert_event: AlertEvent) -> Dict[str, Any]:
        """Traitement par fenêtres"""
        # Détermination de la fenêtre appropriée
        window = await self._get_or_create_window(alert_event.timestamp)
        
        # Ajout de l'événement à la fenêtre
        window.events.append(alert_event)
        
        # Traitement si fenêtre complète
        if self._is_window_complete(window):
            await self._process_complete_window(window)
        
        return {
            'event_id': alert_event.id,
            'processing_mode': 'windowed',
            'window_id': window.window_id,
            'processed_at': datetime.now().isoformat()
        }
    
    async def _event_processing_loop(self):
        """Boucle principale de traitement d'événements"""
        while self.is_running:
            try:
                if self.event_buffer:
                    # Traitement en batch pour optimisation
                    batch_size = min(len(self.event_buffer), 
                                   self.config.get('batch_size', 100))
                    
                    events_to_process = []
                    for _ in range(batch_size):
                        if self.event_buffer:
                            events_to_process.append(self.event_buffer.popleft())
                    
                    if events_to_process:
                        await self._process_event_batch(events_to_process)
                
                # Attente courte pour éviter la surconsommation CPU
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de traitement: {e}")
                await asyncio.sleep(1)
    
    async def _process_event_batch(self, events: List[AlertEvent]):
        """Traitement d'un lot d'événements"""
        try:
            # Traitement parallèle des événements
            tasks = [self.process_event(event) for event in events]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Agrégation des résultats
            successful_results = [r for r in results if not isinstance(r, Exception)]
            error_count = len(results) - len(successful_results)
            
            if error_count > 0:
                logger.warning(f"{error_count} erreurs sur {len(events)} événements traités")
            
            # Mise à jour des métriques de débit
            await self._update_throughput_metrics()
            
        except Exception as e:
            logger.error(f"Erreur traitement batch: {e}")
    
    async def _window_management_loop(self):
        """Boucle de gestion des fenêtres"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Vérification des fenêtres expirées
                expired_windows = [
                    window_id for window_id, window in self.active_windows.items()
                    if current_time >= window.end_time
                ]
                
                # Traitement des fenêtres expirées
                for window_id in expired_windows:
                    window = self.active_windows.pop(window_id)
                    await self._process_complete_window(window)
                
                # Création de nouvelles fenêtres si nécessaire
                if self.processing_mode == ProcessingMode.SLIDING_WINDOW:
                    await self._create_sliding_windows(current_time)
                
                # Mise à jour des métriques
                self.active_windows_gauge.set(len(self.active_windows))
                
                # Attente basée sur l'intervalle de slide
                await asyncio.sleep(self.slide_interval.total_seconds())
                
            except Exception as e:
                logger.error(f"Erreur gestion fenêtres: {e}")
                await asyncio.sleep(1)
    
    async def _metrics_update_loop(self):
        """Boucle de mise à jour des métriques"""
        while self.is_running:
            try:
                # Calcul du débit
                current_time = datetime.now()
                recent_events = [
                    t for t in self.rate_limiter 
                    if (current_time - t).total_seconds() <= 60
                ]
                
                events_per_minute = len(recent_events)
                self.processing_metrics.events_per_second = events_per_minute / 60.0
                self.throughput_gauge.set(self.processing_metrics.events_per_second)
                
                # Persistence des métriques
                await self._persist_metrics()
                
                await asyncio.sleep(30)  # Mise à jour toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"Erreur mise à jour métriques: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self):
        """Boucle de vérification de santé"""
        while self.is_running:
            try:
                # Vérification des connexions
                await self._check_connections_health()
                
                # Vérification de la charge
                await self._check_system_load()
                
                # Auto-scaling si nécessaire
                await self._auto_scale_if_needed()
                
                await asyncio.sleep(60)  # Vérification toutes les minutes
                
            except Exception as e:
                logger.error(f"Erreur health check: {e}")
                await asyncio.sleep(10)
    
    # Méthodes utilitaires
    
    def _generate_window_id(self, timestamp: datetime) -> str:
        """Génération d'ID de fenêtre"""
        window_start = timestamp.replace(
            second=timestamp.second - (timestamp.second % self.window_size.seconds),
            microsecond=0
        )
        return f"window_{window_start.isoformat()}"
    
    async def _get_or_create_window(self, timestamp: datetime) -> StreamingWindow:
        """Récupération ou création de fenêtre"""
        window_id = self._generate_window_id(timestamp)
        
        if window_id not in self.active_windows:
            window_start = timestamp.replace(
                second=timestamp.second - (timestamp.second % self.window_size.seconds),
                microsecond=0
            )
            
            window = StreamingWindow(
                window_id=window_id,
                start_time=window_start,
                end_time=window_start + self.window_size
            )
            
            self.active_windows[window_id] = window
        
        return self.active_windows[window_id]
    
    def _is_window_complete(self, window: StreamingWindow) -> bool:
        """Vérification si une fenêtre est complète"""
        return datetime.now() >= window.end_time
    
    async def _process_complete_window(self, window: StreamingWindow):
        """Traitement d'une fenêtre complète"""
        try:
            if not window.events:
                return
            
            # Agrégation des événements
            aggregated_data = await self._aggregate_window_events(window)
            
            # Analyse ML sur l'agrégation
            ml_results = await self._analyze_window_ml(window, aggregated_data)
            
            # Publication des résultats
            await self._publish_window_results(window, aggregated_data, ml_results)
            
            # Persistance
            await self._persist_window_results(window, aggregated_data)
            
            window.is_complete = True
            
            logger.debug(f"Fenêtre {window.window_id} traitée: {len(window.events)} événements")
            
        except Exception as e:
            logger.error(f"Erreur traitement fenêtre {window.window_id}: {e}")
    
    async def _aggregate_window_events(self, window: StreamingWindow) -> Dict[str, Any]:
        """Agrégation des événements d'une fenêtre"""
        events = window.events
        
        aggregation = {
            'window_id': window.window_id,
            'time_range': {
                'start': window.start_time.isoformat(),
                'end': window.end_time.isoformat()
            },
            'event_count': len(events),
            'severity_distribution': {},
            'service_distribution': {},
            'average_metrics': {},
            'anomaly_events': [],
            'correlation_clusters': []
        }
        
        # Distribution par gravité
        for event in events:
            severity = event.severity.value
            aggregation['severity_distribution'][severity] = \
                aggregation['severity_distribution'].get(severity, 0) + 1
        
        # Distribution par service
        for event in events:
            service = event.service
            aggregation['service_distribution'][service] = \
                aggregation['service_distribution'].get(service, 0) + 1
        
        # Métriques moyennes
        metric_sums = defaultdict(list)
        for event in events:
            for metric, value in event.metrics.custom_metrics.items():
                metric_sums[metric].append(value)
        
        for metric, values in metric_sums.items():
            aggregation['average_metrics'][metric] = np.mean(values)
        
        return aggregation
    
    def _convert_to_time_series_point(self, alert_event: AlertEvent):
        """Conversion vers point de série temporelle"""
        return {
            'timestamp': alert_event.timestamp,
            'value': alert_event.current_value or 0.0,
            'features': alert_event.metrics.custom_metrics,
            'metadata': {
                'service': alert_event.service,
                'severity': alert_event.severity.value
            }
        }
    
    def _convert_to_correlation_event(self, alert_event: AlertEvent):
        """Conversion vers événement de corrélation"""
        from ..correlation_analyzer import CorrelationEvent
        return CorrelationEvent(
            id=alert_event.id,
            timestamp=alert_event.timestamp,
            service=alert_event.service,
            component=alert_event.component or 'unknown',
            severity=alert_event.severity.value,
            message=alert_event.message,
            labels=alert_event.labels,
            metrics=alert_event.metrics.custom_metrics
        )
    
    async def _notify_anomaly(self, alert_event: AlertEvent, anomaly_result):
        """Notification d'anomalie détectée"""
        for handler in self.anomaly_handlers:
            try:
                await handler(alert_event, anomaly_result)
            except Exception as e:
                logger.error(f"Erreur handler anomalie: {e}")
    
    async def _notify_correlation(self, alert_event: AlertEvent, correlation_result):
        """Notification de corrélation détectée"""
        for handler in self.correlation_handlers:
            try:
                await handler(alert_event, correlation_result)
            except Exception as e:
                logger.error(f"Erreur handler corrélation: {e}")
    
    async def _publish_results(self, alert_event: AlertEvent, result: Dict[str, Any]):
        """Publication des résultats"""
        if self.kafka_producer:
            try:
                topic = self.config.get('output_topic', 'alert-analytics-results')
                self.kafka_producer.send(topic, value=result, key=alert_event.id)
            except Exception as e:
                logger.error(f"Erreur publication Kafka: {e}")
    
    async def _cleanup_connections(self):
        """Nettoyage des connexions"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.db_pool:
                await self.db_pool.close()
            
            if self.kafka_producer:
                self.kafka_producer.close()
            
            if self.anomaly_detector:
                await self.anomaly_detector.close()
            
            if self.correlation_analyzer:
                await self.correlation_analyzer.close()
                
        except Exception as e:
            logger.error(f"Erreur nettoyage connexions: {e}")
    
    # Interface publique pour configuration
    
    def add_event_handler(self, handler: Callable):
        """Ajout d'un gestionnaire d'événements"""
        self.event_handlers.append(handler)
    
    def add_anomaly_handler(self, handler: Callable):
        """Ajout d'un gestionnaire d'anomalies"""
        self.anomaly_handlers.append(handler)
    
    def add_correlation_handler(self, handler: Callable):
        """Ajout d'un gestionnaire de corrélations"""
        self.correlation_handlers.append(handler)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Récupération des statistiques de traitement"""
        return {
            'events_processed': self.processing_metrics.events_processed,
            'events_per_second': self.processing_metrics.events_per_second,
            'processing_latency_ms': self.processing_metrics.processing_latency_ms,
            'error_count': self.processing_metrics.error_count,
            'active_windows': len(self.active_windows),
            'buffer_size': len(self.event_buffer),
            'is_running': self.is_running,
            'last_processing_time': self.processing_metrics.last_processing_time.isoformat() if self.processing_metrics.last_processing_time else None
        }


# Factory pour création d'instance
async def create_stream_processor(config: Dict[str, Any]) -> AlertStreamProcessor:
    """Factory pour créer et initialiser le processeur de streaming"""
    processor = AlertStreamProcessor(config)
    await processor.initialize()
    return processor
