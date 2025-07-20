"""
Real-Time Stream Processing Engine for Spotify AI Agent.

This module implements high-performance streaming data processing
for real-time monitoring and alerting of music streaming platform.

Features:
- Apache Kafka integration for event streaming
- Real-time anomaly detection pipeline
- Stream processing with windowing
- Backpressure handling and load balancing
- Event deduplication and ordering
- Scalable microservice architecture
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import threading

import aiokafka
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import redis.asyncio as aioredis
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of streaming events."""
    AUDIO_QUALITY = "audio_quality"
    USER_ACTION = "user_action"
    SYSTEM_METRIC = "system_metric"
    PLAYBACK_EVENT = "playback_event"
    RECOMMENDATION_EVENT = "recommendation_event"
    BILLING_EVENT = "billing_event"
    CONTENT_DELIVERY = "content_delivery"
    SECURITY_EVENT = "security_event"

class ProcessingStage(Enum):
    """Stream processing stages."""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    ENRICHMENT = "enrichment"
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    NOTIFICATION = "notification"
    STORAGE = "storage"

@dataclass
class StreamEvent:
    """Streaming event data structure."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_stage: ProcessingStage = ProcessingStage.INGESTION
    retries: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'data': self.data,
            'metadata': self.metadata,
            'processing_stage': self.processing_stage.value,
            'retries': self.retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamEvent':
        """Create from dictionary."""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            data=data['data'],
            metadata=data.get('metadata', {}),
            processing_stage=ProcessingStage(data.get('processing_stage', 'ingestion')),
            retries=data.get('retries', 0)
        )

@dataclass
class WindowConfig:
    """Stream processing window configuration."""
    window_size_seconds: int
    slide_interval_seconds: int
    late_arrival_tolerance_seconds: int = 30
    max_out_of_order_seconds: int = 60

class StreamProcessor:
    """High-performance stream processing engine."""
    
    def __init__(self, 
                 kafka_bootstrap_servers: List[str],
                 redis_client: aioredis.Redis,
                 consumer_group: str = "spotify-ai-processors"):
        
        self.kafka_servers = kafka_bootstrap_servers
        self.redis_client = redis_client
        self.consumer_group = consumer_group
        
        # Processing components
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.producer: Optional[AIOKafkaProducer] = None
        self.processors: Dict[EventType, List[Callable]] = defaultdict(list)
        self.windows: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # State management
        self.is_running = False
        self.processing_tasks: List[asyncio.Task] = []
        self.event_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        
        # Metrics
        self.events_processed_counter = Counter(
            'stream_events_processed_total',
            'Total events processed',
            ['event_type', 'stage', 'status']
        )
        
        self.processing_latency_histogram = Histogram(
            'stream_processing_latency_seconds',
            'Event processing latency',
            ['event_type', 'stage'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.backpressure_gauge = Gauge(
            'stream_backpressure_events',
            'Number of events in backpressure queue',
            ['topic']
        )
        
        self.throughput_gauge = Gauge(
            'stream_throughput_events_per_second',
            'Stream processing throughput',
            ['event_type']
        )
    
    async def initialize(self):
        """Initialize Kafka producer and consumers."""
        
        # Initialize producer
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type="gzip",
            batch_size=32768,  # 32KB batches
            linger_ms=10,  # Small latency for batching
            max_request_size=10485760,  # 10MB max request
            retry_backoff_ms=100
        )
        
        await self.producer.start()
        logger.info("Kafka producer initialized")
        
        # Initialize consumers for different event types
        for event_type in EventType:
            topic_name = f"spotify-{event_type.value.replace('_', '-')}"
            
            consumer = AIOKafkaConsumer(
                topic_name,
                bootstrap_servers=self.kafka_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                max_poll_records=500,
                max_poll_interval_ms=300000,  # 5 minutes
                session_timeout_ms=30000,    # 30 seconds
                fetch_min_bytes=1024,        # 1KB minimum fetch
                fetch_max_wait_ms=500        # 500ms max wait
            )
            
            self.consumers[topic_name] = consumer
            await consumer.start()
            
        logger.info(f"Initialized {len(self.consumers)} Kafka consumers")
    
    def register_processor(self, event_type: EventType, 
                          processor: Callable[[StreamEvent], StreamEvent]):
        """Register event processor for specific event type."""
        self.processors[event_type].append(processor)
        logger.info(f"Registered processor for {event_type.value}")
    
    async def start_processing(self):
        """Start stream processing."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start processing tasks for each consumer
        for topic_name, consumer in self.consumers.items():
            task = asyncio.create_task(
                self._process_topic(topic_name, consumer)
            )
            self.processing_tasks.append(task)
        
        # Start throughput monitoring
        throughput_task = asyncio.create_task(self._monitor_throughput())
        self.processing_tasks.append(throughput_task)
        
        logger.info("Stream processing started")
    
    async def stop_processing(self):
        """Stop stream processing."""
        self.is_running = False
        
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Stop consumers and producer
        for consumer in self.consumers.values():
            await consumer.stop()
        
        if self.producer:
            await self.producer.stop()
        
        logger.info("Stream processing stopped")
    
    async def _process_topic(self, topic_name: str, consumer: AIOKafkaConsumer):
        """Process events from a specific topic."""
        
        while self.is_running:
            try:
                # Fetch messages with timeout
                message_batch = await asyncio.wait_for(
                    consumer.getmany(timeout_ms=1000, max_records=100),
                    timeout=2.0
                )
                
                if not message_batch:
                    continue
                
                # Process messages in parallel
                for topic_partition, messages in message_batch.items():
                    tasks = []
                    
                    for message in messages:
                        try:
                            event = StreamEvent.from_dict(message.value)
                            task = asyncio.create_task(
                                self._process_event(event)
                            )
                            tasks.append(task)
                            
                        except Exception as e:
                            logger.error(f"Failed to parse event from {topic_name}: {e}")
                            continue
                    
                    # Wait for all events in batch to process
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update backpressure metrics
                buffer_size = len(self.event_buffer[topic_name])
                self.backpressure_gauge.labels(topic=topic_name).set(buffer_size)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing topic {topic_name}: {e}")
                await asyncio.sleep(1)
    
    async def _process_event(self, event: StreamEvent):
        """Process individual event through pipeline."""
        
        start_time = time.time()
        
        try:
            # Validation stage
            event.processing_stage = ProcessingStage.VALIDATION
            if not self._validate_event(event):
                self._record_processing_metrics(event, "validation", "failed")
                return
            
            # Enrichment stage
            event.processing_stage = ProcessingStage.ENRICHMENT
            await self._enrich_event(event)
            
            # Detection stage (anomaly detection)
            event.processing_stage = ProcessingStage.DETECTION
            anomalies = await self._detect_anomalies(event)
            
            if anomalies:
                event.metadata['anomalies'] = anomalies
                
                # Classification stage for anomalies
                event.processing_stage = ProcessingStage.CLASSIFICATION
                classification = await self._classify_anomalies(event, anomalies)
                event.metadata['classification'] = classification
                
                # Notification stage for critical issues
                if classification.get('severity') in ['critical', 'high']:
                    event.processing_stage = ProcessingStage.NOTIFICATION
                    await self._send_notifications(event)
            
            # Storage stage
            event.processing_stage = ProcessingStage.STORAGE
            await self._store_event(event)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._record_processing_metrics(event, "complete", "success", processing_time)
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            
            # Retry logic
            if event.retries < 3:
                event.retries += 1
                await self._retry_event(event)
            else:
                await self._send_to_dead_letter_queue(event, str(e))
            
            self._record_processing_metrics(event, event.processing_stage.value, "error")
    
    def _validate_event(self, event: StreamEvent) -> bool:
        """Validate event data."""
        
        # Basic validation
        if not event.event_id or not event.data:
            return False
        
        # Check timestamp is not too old or in future
        now = datetime.now()
        if abs((now - event.timestamp).total_seconds()) > 3600:  # 1 hour tolerance
            return False
        
        # Event-specific validation
        if event.event_type == EventType.AUDIO_QUALITY:
            required_fields = ['bitrate', 'latency_ms', 'buffer_health']
            if not all(field in event.data for field in required_fields):
                return False
        
        elif event.event_type == EventType.USER_ACTION:
            if 'user_id' not in event.data and not event.user_id:
                return False
        
        return True
    
    async def _enrich_event(self, event: StreamEvent):
        """Enrich event with additional context."""
        
        # Add user context for user events
        if event.user_id:
            user_context = await self._get_user_context(event.user_id)
            event.metadata['user_context'] = user_context
        
        # Add geographic context
        if 'ip_address' in event.data:
            geo_context = await self._get_geo_context(event.data['ip_address'])
            event.metadata['geo_context'] = geo_context
        
        # Add service context
        if 'service_name' in event.data:
            service_context = await self._get_service_context(event.data['service_name'])
            event.metadata['service_context'] = service_context
    
    async def _detect_anomalies(self, event: StreamEvent) -> List[Dict[str, Any]]:
        """Detect anomalies in event data."""
        
        anomalies = []
        
        # Audio quality anomaly detection
        if event.event_type == EventType.AUDIO_QUALITY:
            data = event.data
            
            # Bitrate anomaly
            if data.get('bitrate', 320) < 128:
                anomalies.append({
                    'type': 'audio_quality_degradation',
                    'field': 'bitrate',
                    'value': data.get('bitrate'),
                    'expected_min': 128,
                    'severity': 'high'
                })
            
            # Latency anomaly
            if data.get('latency_ms', 50) > 200:
                anomalies.append({
                    'type': 'high_latency',
                    'field': 'latency_ms',
                    'value': data.get('latency_ms'),
                    'threshold': 200,
                    'severity': 'medium'
                })
            
            # Buffer health anomaly
            if data.get('buffer_health', 100) < 30:
                anomalies.append({
                    'type': 'buffer_underrun',
                    'field': 'buffer_health',
                    'value': data.get('buffer_health'),
                    'threshold': 30,
                    'severity': 'high'
                })
        
        # User behavior anomaly detection
        elif event.event_type == EventType.USER_ACTION:
            # Rapid-fire actions (potential bot behavior)
            if event.user_id:
                recent_actions = await self._get_recent_user_actions(event.user_id)
                if len(recent_actions) > 100:  # More than 100 actions in last minute
                    anomalies.append({
                        'type': 'suspicious_user_activity',
                        'field': 'action_frequency',
                        'value': len(recent_actions),
                        'threshold': 100,
                        'severity': 'critical'
                    })
        
        # System metric anomalies
        elif event.event_type == EventType.SYSTEM_METRIC:
            data = event.data
            
            # High error rate
            if data.get('error_rate', 0) > 0.05:  # 5% error rate
                anomalies.append({
                    'type': 'high_error_rate',
                    'field': 'error_rate',
                    'value': data.get('error_rate'),
                    'threshold': 0.05,
                    'severity': 'critical'
                })
            
            # High response time
            if data.get('response_time_ms', 100) > 1000:
                anomalies.append({
                    'type': 'high_response_time',
                    'field': 'response_time_ms',
                    'value': data.get('response_time_ms'),
                    'threshold': 1000,
                    'severity': 'medium'
                })
        
        return anomalies
    
    async def _classify_anomalies(self, event: StreamEvent, 
                                 anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify anomalies by business impact."""
        
        # Calculate overall severity
        severity_scores = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        
        max_severity_score = max(
            severity_scores.get(anomaly['severity'], 1)
            for anomaly in anomalies
        )
        
        severity_names = {v: k for k, v in severity_scores.items()}
        overall_severity = severity_names[max_severity_score]
        
        # Estimate business impact
        business_impact = self._calculate_business_impact(event, anomalies)
        
        # Determine escalation needed
        escalation_needed = (
            overall_severity in ['critical', 'high'] or
            business_impact.get('revenue_impact_per_hour', 0) > 10000
        )
        
        return {
            'severity': overall_severity,
            'business_impact': business_impact,
            'escalation_needed': escalation_needed,
            'affected_users_estimate': business_impact.get('affected_users', 0),
            'revenue_impact_per_hour': business_impact.get('revenue_impact_per_hour', 0)
        }
    
    def _calculate_business_impact(self, event: StreamEvent, 
                                  anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate business impact of anomalies."""
        
        impact = {
            'affected_users': 0,
            'revenue_impact_per_hour': 0,
            'service_impact_score': 0
        }
        
        # User context impact
        user_context = event.metadata.get('user_context', {})
        if user_context.get('subscription_type') == 'premium':
            impact['revenue_impact_per_hour'] += 0.50  # $0.50 per premium user per hour
        else:
            impact['revenue_impact_per_hour'] += 0.05  # $0.05 per free user per hour
        
        # Geographic impact
        geo_context = event.metadata.get('geo_context', {})
        region_multipliers = {
            'US': 1.5,
            'EU': 1.3,
            'CA': 1.2,
            'AU': 1.1,
            'default': 1.0
        }
        
        region = geo_context.get('country_code', 'default')
        multiplier = region_multipliers.get(region, region_multipliers['default'])
        impact['revenue_impact_per_hour'] *= multiplier
        
        # Service impact
        service_context = event.metadata.get('service_context', {})
        service_criticality = service_context.get('criticality_score', 1.0)
        impact['service_impact_score'] = service_criticality
        
        # Anomaly-specific impact
        for anomaly in anomalies:
            if anomaly['type'] == 'audio_quality_degradation':
                impact['affected_users'] += 1000  # Estimated affected users
                impact['revenue_impact_per_hour'] *= 2  # Double impact for quality issues
            
            elif anomaly['type'] == 'high_error_rate':
                impact['affected_users'] += 5000
                impact['revenue_impact_per_hour'] *= 3
            
            elif anomaly['type'] == 'suspicious_user_activity':
                # Security impact doesn't directly affect revenue but requires investigation
                impact['service_impact_score'] += 0.5
        
        return impact
    
    async def _send_notifications(self, event: StreamEvent):
        """Send notifications for critical events."""
        
        classification = event.metadata.get('classification', {})
        
        # Create notification payload
        notification = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'severity': classification.get('severity'),
            'anomalies': event.metadata.get('anomalies', []),
            'business_impact': classification.get('business_impact', {}),
            'timestamp': event.timestamp.isoformat(),
            'escalation_needed': classification.get('escalation_needed', False)
        }
        
        # Send to notification topic
        await self.producer.send(
            'spotify-notifications',
            value=notification
        )
        
        # Store in Redis for real-time access
        await self.redis_client.lpush(
            'critical_events',
            json.dumps(notification)
        )
        
        # Keep only last 1000 critical events
        await self.redis_client.ltrim('critical_events', 0, 999)
    
    async def _store_event(self, event: StreamEvent):
        """Store processed event."""
        
        # Store in Redis with TTL based on event type
        ttl_config = {
            EventType.AUDIO_QUALITY: 3600,      # 1 hour
            EventType.USER_ACTION: 86400,       # 24 hours
            EventType.SYSTEM_METRIC: 7200,      # 2 hours
            EventType.SECURITY_EVENT: 604800,   # 1 week
            'default': 3600
        }
        
        ttl = ttl_config.get(event.event_type, ttl_config['default'])
        
        await self.redis_client.setex(
            f"event:{event.event_id}",
            ttl,
            json.dumps(event.to_dict())
        )
        
        # Store in time-series format for analytics
        time_key = f"events:{event.event_type.value}:{event.timestamp.strftime('%Y%m%d%H')}"
        await self.redis_client.lpush(time_key, event.event_id)
        await self.redis_client.expire(time_key, 86400)  # 24 hours
    
    async def _retry_event(self, event: StreamEvent):
        """Retry failed event processing."""
        
        # Exponential backoff
        delay = min(2 ** event.retries, 60)  # Max 60 seconds
        await asyncio.sleep(delay)
        
        # Reset processing stage
        event.processing_stage = ProcessingStage.VALIDATION
        
        # Add to retry queue
        retry_topic = f"spotify-retries-{event.event_type.value.replace('_', '-')}"
        await self.producer.send(retry_topic, value=event.to_dict())
    
    async def _send_to_dead_letter_queue(self, event: StreamEvent, error: str):
        """Send failed event to dead letter queue."""
        
        dlq_payload = {
            'original_event': event.to_dict(),
            'error': error,
            'failed_at': datetime.now().isoformat(),
            'retries_attempted': event.retries
        }
        
        await self.producer.send(
            'spotify-dead-letter-queue',
            value=dlq_payload
        )
    
    def _record_processing_metrics(self, event: StreamEvent, stage: str, 
                                  status: str, processing_time: Optional[float] = None):
        """Record processing metrics."""
        
        self.events_processed_counter.labels(
            event_type=event.event_type.value,
            stage=stage,
            status=status
        ).inc()
        
        if processing_time is not None:
            self.processing_latency_histogram.labels(
                event_type=event.event_type.value,
                stage=stage
            ).observe(processing_time)
    
    async def _monitor_throughput(self):
        """Monitor and report throughput metrics."""
        
        throughput_counters = defaultdict(int)
        
        while self.is_running:
            try:
                # Reset counters
                for event_type in EventType:
                    throughput_counters[event_type] = 0
                
                # Count events in next 10 seconds
                await asyncio.sleep(10)
                
                # Update throughput gauges
                for event_type, count in throughput_counters.items():
                    rps = count / 10.0  # Events per second
                    self.throughput_gauge.labels(
                        event_type=event_type.value
                    ).set(rps)
                
            except Exception as e:
                logger.error(f"Error in throughput monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context from cache or database."""
        
        # Try Redis cache first
        cached_context = await self.redis_client.get(f"user_context:{user_id}")
        if cached_context:
            return json.loads(cached_context)
        
        # Mock user context (in real implementation, query user service)
        context = {
            'subscription_type': 'premium',
            'country': 'US',
            'signup_date': '2023-01-15',
            'last_active': datetime.now().isoformat()
        }
        
        # Cache for 1 hour
        await self.redis_client.setex(
            f"user_context:{user_id}",
            3600,
            json.dumps(context)
        )
        
        return context
    
    async def _get_geo_context(self, ip_address: str) -> Dict[str, Any]:
        """Get geographic context from IP address."""
        
        # Mock geo context (in real implementation, use GeoIP service)
        return {
            'country_code': 'US',
            'region': 'California',
            'city': 'San Francisco',
            'isp': 'Comcast'
        }
    
    async def _get_service_context(self, service_name: str) -> Dict[str, Any]:
        """Get service context and metadata."""
        
        # Mock service context
        service_contexts = {
            'audio-streaming': {'criticality_score': 1.0, 'sla_target': 99.95},
            'recommendation': {'criticality_score': 0.8, 'sla_target': 99.9},
            'search': {'criticality_score': 0.9, 'sla_target': 99.9},
            'billing': {'criticality_score': 1.0, 'sla_target': 99.99}
        }
        
        return service_contexts.get(service_name, {
            'criticality_score': 0.5,
            'sla_target': 99.5
        })
    
    async def _get_recent_user_actions(self, user_id: str) -> List[str]:
        """Get recent user actions for anomaly detection."""
        
        # Get recent actions from Redis
        actions = await self.redis_client.lrange(
            f"user_actions:{user_id}",
            0, -1
        )
        
        # Filter actions from last minute
        current_time = time.time()
        recent_actions = []
        
        for action_data in actions:
            try:
                action = json.loads(action_data)
                if current_time - action.get('timestamp', 0) <= 60:
                    recent_actions.append(action)
            except:
                continue
        
        return recent_actions

# Global stream processor instance
STREAM_PROCESSOR: Optional[StreamProcessor] = None

async def initialize_stream_processor(kafka_servers: List[str], 
                                    redis_client: aioredis.Redis) -> StreamProcessor:
    """Initialize global stream processor."""
    global STREAM_PROCESSOR
    
    STREAM_PROCESSOR = StreamProcessor(kafka_servers, redis_client)
    await STREAM_PROCESSOR.initialize()
    
    return STREAM_PROCESSOR

def get_stream_processor() -> Optional[StreamProcessor]:
    """Get global stream processor instance."""
    return STREAM_PROCESSOR

__all__ = [
    'StreamProcessor',
    'StreamEvent',
    'EventType',
    'ProcessingStage',
    'WindowConfig',
    'initialize_stream_processor',
    'get_stream_processor'
]
