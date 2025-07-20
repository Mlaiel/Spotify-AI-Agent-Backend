#!/usr/bin/env python3
"""
Real-Time Data Pipeline for Spotify AI Agent Analytics
====================================================

Ultra-sophisticated real-time data processing pipeline with advanced stream processing,
event-driven architecture, and enterprise-grade reliability.

Author: Fahed Mlaiel
Roles: Lead Dev + Architecte IA, DBA & Data Engineer, Architecte Microservices
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
import aioredis
import aiokafka
import aiofiles
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import uvloop

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class StreamEvent:
    """Represents a streaming event in the data pipeline."""
    event_id: str
    tenant_id: str
    user_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class ProcessingResult:
    """Result of stream processing operation."""
    success: bool
    processed_events: int
    failed_events: int
    processing_time: float
    errors: List[str]

class StreamProcessor:
    """Real-time stream processing engine."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kafka_consumer = None
        self.kafka_producer = None
        self.redis_client = None
        self.event_handlers = {}
        self.metrics = self._setup_metrics()
        self.is_running = False
        self.processing_queue = asyncio.Queue(maxsize=10000)
        self.dead_letter_queue = asyncio.Queue(maxsize=1000)
        
    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup Prometheus metrics for monitoring."""
        return {
            'events_processed_total': Counter(
                'spotify_stream_events_processed_total',
                'Total events processed',
                ['tenant_id', 'event_type', 'status']
            ),
            'processing_duration_seconds': Histogram(
                'spotify_stream_processing_duration_seconds',
                'Time spent processing events',
                ['tenant_id', 'event_type']
            ),
            'queue_size': Gauge(
                'spotify_stream_queue_size',
                'Current queue size',
                ['queue_type']
            ),
            'processing_errors_total': Counter(
                'spotify_stream_processing_errors_total',
                'Total processing errors',
                ['tenant_id', 'event_type', 'error_type']
            )
        }

    async def initialize(self) -> None:
        """Initialize the stream processor."""
        try:
            # Initialize Kafka consumer
            self.kafka_consumer = aiokafka.AIOKafkaConsumer(
                *self.config['kafka']['topics'],
                bootstrap_servers=self.config['kafka']['bootstrap_servers'],
                group_id=self.config['kafka']['group_id'],
                auto_offset_reset='latest',
                enable_auto_commit=False,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            
            # Initialize Kafka producer for output
            self.kafka_producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers=self.config['kafka']['bootstrap_servers'],
                value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8')
            )
            
            # Initialize Redis for state management
            self.redis_client = aioredis.from_url(
                f"redis://{self.config['redis']['host']}:{self.config['redis']['port']}",
                db=self.config['redis']['db'],
                decode_responses=True
            )
            
            await self.kafka_consumer.start()
            await self.kafka_producer.start()
            
            # Start Prometheus metrics server
            start_http_server(self.config.get('metrics_port', 8000))
            
            logger.info("Stream processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing stream processor: {e}")
            raise

    def register_event_handler(self, event_type: str, handler: Callable[[StreamEvent], Any]) -> None:
        """Register an event handler for specific event types."""
        self.event_handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")

    async def start_processing(self) -> None:
        """Start the main processing loop."""
        if self.is_running:
            return
            
        self.is_running = True
        logger.info("Starting stream processing")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._consume_events()),
            asyncio.create_task(self._process_events()),
            asyncio.create_task(self._handle_dead_letter_queue()),
            asyncio.create_task(self._update_metrics()),
            asyncio.create_task(self._health_check())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in processing tasks: {e}")
        finally:
            self.is_running = False

    async def stop_processing(self) -> None:
        """Stop stream processing gracefully."""
        logger.info("Stopping stream processing")
        self.is_running = False
        
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        if self.kafka_producer:
            await self.kafka_producer.stop()
        if self.redis_client:
            await self.redis_client.close()

    async def _consume_events(self) -> None:
        """Consume events from Kafka topics."""
        while self.is_running:
            try:
                # Get next message batch
                msg_pack = await self.kafka_consumer.getmany(
                    timeout_ms=1000,
                    max_records=self.config.get('batch_size', 100)
                )
                
                for tp, messages in msg_pack.items():
                    for message in messages:
                        try:
                            # Parse stream event
                            event = self._parse_event(message)
                            
                            # Add to processing queue
                            await self.processing_queue.put(event)
                            
                            # Update metrics
                            self.metrics['queue_size'].labels(queue_type='processing').set(
                                self.processing_queue.qsize()
                            )
                            
                        except Exception as e:
                            logger.error(f"Error parsing event: {e}")
                            self.metrics['processing_errors_total'].labels(
                                tenant_id='unknown',
                                event_type='unknown',
                                error_type='parse_error'
                            ).inc()
                
                # Commit offsets after successful processing
                await self.kafka_consumer.commit()
                
            except Exception as e:
                logger.error(f"Error consuming events: {e}")
                await asyncio.sleep(1)

    def _parse_event(self, message: aiokafka.ConsumerRecord) -> StreamEvent:
        """Parse Kafka message into StreamEvent."""
        data = message.value
        
        return StreamEvent(
            event_id=data.get('event_id', f"event_{message.offset}"),
            tenant_id=data.get('tenant_id', 'unknown'),
            user_id=data.get('user_id', 'unknown'),
            event_type=data.get('event_type', 'unknown'),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            data=data.get('data', {}),
            metadata={
                'topic': message.topic,
                'partition': message.partition,
                'offset': message.offset,
                'headers': dict(message.headers) if message.headers else {}
            }
        )

    async def _process_events(self) -> None:
        """Process events from the queue."""
        batch_size = self.config.get('processing_batch_size', 50)
        
        while self.is_running:
            try:
                # Collect batch of events
                batch = []
                for _ in range(batch_size):
                    try:
                        event = await asyncio.wait_for(
                            self.processing_queue.get(), timeout=1.0
                        )
                        batch.append(event)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self._process_event_batch(batch)
                    
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)

    async def _process_event_batch(self, events: List[StreamEvent]) -> None:
        """Process a batch of events."""
        start_time = datetime.now()
        processed_count = 0
        failed_count = 0
        
        try:
            # Group events by type for efficient processing
            events_by_type = {}
            for event in events:
                if event.event_type not in events_by_type:
                    events_by_type[event.event_type] = []
                events_by_type[event.event_type].append(event)
            
            # Process each event type
            for event_type, type_events in events_by_type.items():
                try:
                    await self._process_events_by_type(event_type, type_events)
                    processed_count += len(type_events)
                    
                except Exception as e:
                    logger.error(f"Error processing events of type {event_type}: {e}")
                    failed_count += len(type_events)
                    
                    # Add failed events to dead letter queue
                    for event in type_events:
                        await self._add_to_dead_letter_queue(event, str(e))
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            for event in events:
                self.metrics['events_processed_total'].labels(
                    tenant_id=event.tenant_id,
                    event_type=event.event_type,
                    status='success' if processed_count > 0 else 'failed'
                ).inc()
                
                self.metrics['processing_duration_seconds'].labels(
                    tenant_id=event.tenant_id,
                    event_type=event.event_type
                ).observe(processing_time / len(events))
                
        except Exception as e:
            logger.error(f"Error processing event batch: {e}")

    async def _process_events_by_type(self, event_type: str, events: List[StreamEvent]) -> None:
        """Process events of a specific type."""
        try:
            if event_type in self.event_handlers:
                # Use registered handler
                handler = self.event_handlers[event_type]
                await handler(events)
            else:
                # Use default processing
                await self._default_event_processing(events)
                
            # Store processed events
            await self._store_processed_events(events)
            
            # Generate derived events if needed
            await self._generate_derived_events(events)
            
        except Exception as e:
            logger.error(f"Error processing events of type {event_type}: {e}")
            raise

    async def _default_event_processing(self, events: List[StreamEvent]) -> None:
        """Default event processing logic."""
        for event in events:
            try:
                # Basic event validation
                if not self._validate_event(event):
                    raise ValueError(f"Invalid event: {event.event_id}")
                
                # Enrich event data
                await self._enrich_event(event)
                
                # Apply business rules
                await self._apply_business_rules(event)
                
                logger.debug(f"Processed event {event.event_id} of type {event.event_type}")
                
            except Exception as e:
                logger.error(f"Error processing event {event.event_id}: {e}")
                raise

    def _validate_event(self, event: StreamEvent) -> bool:
        """Validate event data."""
        # Basic validation rules
        if not event.event_id or not event.tenant_id:
            return False
            
        if event.timestamp > datetime.now() + timedelta(minutes=5):
            return False  # Future events not allowed
            
        if event.timestamp < datetime.now() - timedelta(days=7):
            return False  # Too old events not allowed
            
        return True

    async def _enrich_event(self, event: StreamEvent) -> None:
        """Enrich event with additional data."""
        try:
            # Add geographic information
            if 'ip_address' in event.data:
                # This would typically call a GeoIP service
                event.data['country'] = 'US'  # Placeholder
                event.data['region'] = 'California'  # Placeholder
            
            # Add user context
            user_context = await self._get_user_context(event.user_id, event.tenant_id)
            event.data.update(user_context)
            
            # Add session information
            session_info = await self._get_session_info(event.user_id, event.tenant_id)
            event.data.update(session_info)
            
        except Exception as e:
            logger.error(f"Error enriching event {event.event_id}: {e}")

    async def _get_user_context(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
        """Get user context from cache or database."""
        try:
            # Try to get from Redis cache first
            cache_key = f"user_context:{tenant_id}:{user_id}"
            cached_context = await self.redis_client.get(cache_key)
            
            if cached_context:
                return json.loads(cached_context)
            
            # Fallback to default context
            context = {
                'user_tier': 'free',
                'registration_date': '2023-01-01',
                'preferred_genres': ['pop', 'rock'],
                'location': 'US'
            }
            
            # Cache for future use
            await self.redis_client.setex(
                cache_key, 
                3600,  # 1 hour TTL
                json.dumps(context)
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {}

    async def _get_session_info(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
        """Get session information."""
        try:
            session_key = f"session:{tenant_id}:{user_id}"
            session_data = await self.redis_client.hgetall(session_key)
            
            if session_data:
                return {
                    'session_id': session_data.get('session_id'),
                    'session_start': session_data.get('session_start'),
                    'device_type': session_data.get('device_type', 'unknown'),
                    'platform': session_data.get('platform', 'unknown')
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {}

    async def _apply_business_rules(self, event: StreamEvent) -> None:
        """Apply business rules to the event."""
        try:
            # Apply tenant-specific rules
            await self._apply_tenant_rules(event)
            
            # Apply user-tier-specific rules
            await self._apply_user_tier_rules(event)
            
            # Apply content rules
            await self._apply_content_rules(event)
            
        except Exception as e:
            logger.error(f"Error applying business rules to event {event.event_id}: {e}")

    async def _apply_tenant_rules(self, event: StreamEvent) -> None:
        """Apply tenant-specific business rules."""
        # Get tenant configuration
        tenant_config = await self._get_tenant_config(event.tenant_id)
        
        # Apply data retention rules
        if tenant_config.get('data_retention_days'):
            event.metadata['ttl'] = tenant_config['data_retention_days'] * 24 * 3600
        
        # Apply rate limiting
        if tenant_config.get('rate_limit'):
            await self._check_rate_limit(event, tenant_config['rate_limit'])

    async def _apply_user_tier_rules(self, event: StreamEvent) -> None:
        """Apply user tier specific rules."""
        user_tier = event.data.get('user_tier', 'free')
        
        if user_tier == 'free':
            # Add advertising data for free users
            event.data['ad_enabled'] = True
            event.data['ad_frequency'] = 'high'
        elif user_tier == 'premium':
            # Premium features
            event.data['high_quality_audio'] = True
            event.data['offline_mode'] = True

    async def _apply_content_rules(self, event: StreamEvent) -> None:
        """Apply content-specific rules."""
        if event.event_type == 'track_play':
            # Apply content filtering
            track_id = event.data.get('track_id')
            if track_id:
                content_rating = await self._get_content_rating(track_id)
                event.data['content_rating'] = content_rating
                
                # Check if content is allowed for user
                user_age = event.data.get('user_age', 18)
                if content_rating == 'explicit' and user_age < 18:
                    event.data['content_blocked'] = True

    async def _get_tenant_config(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant configuration."""
        try:
            config_key = f"tenant_config:{tenant_id}"
            config = await self.redis_client.hgetall(config_key)
            return config or {}
        except Exception:
            return {}

    async def _get_content_rating(self, track_id: str) -> str:
        """Get content rating for a track."""
        # This would typically query a content database
        # For demo purposes, return random rating
        import random
        return random.choice(['clean', 'explicit'])

    async def _check_rate_limit(self, event: StreamEvent, rate_limit: int) -> None:
        """Check rate limiting for tenant."""
        window_key = f"rate_limit:{event.tenant_id}:{datetime.now().strftime('%Y%m%d%H%M')}"
        current_count = await self.redis_client.incr(window_key)
        await self.redis_client.expire(window_key, 60)  # 1 minute window
        
        if current_count > rate_limit:
            raise Exception(f"Rate limit exceeded for tenant {event.tenant_id}")

    async def _store_processed_events(self, events: List[StreamEvent]) -> None:
        """Store processed events for analytics."""
        try:
            # Batch store events in Redis
            pipe = self.redis_client.pipeline()
            
            for event in events:
                event_key = f"processed_events:{event.tenant_id}:{event.event_type}:{event.event_id}"
                event_data = asdict(event)
                
                # Convert datetime to string for JSON serialization
                event_data['timestamp'] = event.timestamp.isoformat()
                
                pipe.hset(event_key, mapping=event_data)
                pipe.expire(event_key, 86400 * 7)  # 7 days TTL
            
            await pipe.execute()
            
        except Exception as e:
            logger.error(f"Error storing processed events: {e}")

    async def _generate_derived_events(self, events: List[StreamEvent]) -> None:
        """Generate derived events based on processed events."""
        try:
            for event in events:
                derived_events = []
                
                if event.event_type == 'track_play':
                    # Generate engagement event
                    if event.data.get('listen_duration', 0) > 30:
                        derived_event = StreamEvent(
                            event_id=f"engagement_{event.event_id}",
                            tenant_id=event.tenant_id,
                            user_id=event.user_id,
                            event_type='user_engagement',
                            timestamp=datetime.now(),
                            data={
                                'original_event_id': event.event_id,
                                'engagement_type': 'track_completion',
                                'track_id': event.data.get('track_id'),
                                'listen_duration': event.data.get('listen_duration')
                            },
                            metadata={'derived_from': event.event_id}
                        )
                        derived_events.append(derived_event)
                
                # Send derived events to output topic
                for derived_event in derived_events:
                    await self._send_to_output_topic(derived_event)
                    
        except Exception as e:
            logger.error(f"Error generating derived events: {e}")

    async def _send_to_output_topic(self, event: StreamEvent) -> None:
        """Send event to output Kafka topic."""
        try:
            event_data = asdict(event)
            event_data['timestamp'] = event.timestamp.isoformat()
            
            await self.kafka_producer.send(
                self.config['kafka']['output_topic'],
                value=event_data,
                key=f"{event.tenant_id}:{event.user_id}".encode('utf-8')
            )
            
        except Exception as e:
            logger.error(f"Error sending event to output topic: {e}")

    async def _add_to_dead_letter_queue(self, event: StreamEvent, error: str) -> None:
        """Add failed event to dead letter queue."""
        try:
            failed_event = {
                'event': asdict(event),
                'error': error,
                'failed_at': datetime.now().isoformat(),
                'retry_count': 0
            }
            
            await self.dead_letter_queue.put(failed_event)
            
            self.metrics['queue_size'].labels(queue_type='dead_letter').set(
                self.dead_letter_queue.qsize()
            )
            
        except Exception as e:
            logger.error(f"Error adding to dead letter queue: {e}")

    async def _handle_dead_letter_queue(self) -> None:
        """Handle events in the dead letter queue."""
        while self.is_running:
            try:
                # Process dead letter events
                failed_event = await asyncio.wait_for(
                    self.dead_letter_queue.get(), timeout=5.0
                )
                
                await self._process_dead_letter_event(failed_event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error handling dead letter queue: {e}")

    async def _process_dead_letter_event(self, failed_event: Dict[str, Any]) -> None:
        """Process an event from the dead letter queue."""
        try:
            retry_count = failed_event.get('retry_count', 0)
            max_retries = self.config.get('max_retries', 3)
            
            if retry_count < max_retries:
                # Attempt to reprocess
                event = StreamEvent(**failed_event['event'])
                
                try:
                    await self._default_event_processing([event])
                    logger.info(f"Successfully reprocessed event {event.event_id}")
                except Exception as e:
                    # Increment retry count and requeue
                    failed_event['retry_count'] = retry_count + 1
                    failed_event['last_retry'] = datetime.now().isoformat()
                    await self.dead_letter_queue.put(failed_event)
            else:
                # Max retries reached, log and persist
                await self._persist_failed_event(failed_event)
                
        except Exception as e:
            logger.error(f"Error processing dead letter event: {e}")

    async def _persist_failed_event(self, failed_event: Dict[str, Any]) -> None:
        """Persist permanently failed events."""
        try:
            # Store in Redis with long TTL for manual review
            event_id = failed_event['event']['event_id']
            tenant_id = failed_event['event']['tenant_id']
            
            failed_key = f"failed_events:{tenant_id}:{event_id}"
            await self.redis_client.hset(failed_key, mapping=failed_event)
            await self.redis_client.expire(failed_key, 86400 * 30)  # 30 days
            
            logger.warning(f"Persisted permanently failed event: {event_id}")
            
        except Exception as e:
            logger.error(f"Error persisting failed event: {e}")

    async def _update_metrics(self) -> None:
        """Update system metrics periodically."""
        while self.is_running:
            try:
                # Update queue sizes
                self.metrics['queue_size'].labels(queue_type='processing').set(
                    self.processing_queue.qsize()
                )
                self.metrics['queue_size'].labels(queue_type='dead_letter').set(
                    self.dead_letter_queue.qsize()
                )
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")

    async def _health_check(self) -> None:
        """Perform periodic health checks."""
        while self.is_running:
            try:
                # Check Kafka connectivity
                if self.kafka_consumer:
                    # Simple connectivity check
                    await self.kafka_consumer.list_consumer_group_offsets()
                
                # Check Redis connectivity
                if self.redis_client:
                    await self.redis_client.ping()
                
                # Check queue health
                if self.processing_queue.qsize() > 5000:
                    logger.warning("Processing queue is getting full")
                
                if self.dead_letter_queue.qsize() > 500:
                    logger.warning("Dead letter queue has many failed events")
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")

async def main():
    """Main entry point for the stream processor."""
    # Configuration
    config = {
        'kafka': {
            'bootstrap_servers': ['localhost:9092'],
            'topics': ['user_events', 'track_events', 'system_events'],
            'group_id': 'spotify_analytics_processor',
            'output_topic': 'processed_events'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        },
        'batch_size': 100,
        'processing_batch_size': 50,
        'max_retries': 3,
        'metrics_port': 8000
    }
    
    # Set up signal handlers for graceful shutdown
    processor = StreamProcessor(config)
    
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(processor.stop_processing())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and start processing
    try:
        await processor.initialize()
        
        # Register custom event handlers
        processor.register_event_handler('track_play', handle_track_play_events)
        processor.register_event_handler('user_signup', handle_user_signup_events)
        processor.register_event_handler('subscription_change', handle_subscription_events)
        
        # Start processing
        await processor.start_processing()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

async def handle_track_play_events(events: List[StreamEvent]) -> None:
    """Handle track play events specifically."""
    for event in events:
        # Extract relevant data
        track_id = event.data.get('track_id')
        user_id = event.user_id
        listen_duration = event.data.get('listen_duration', 0)
        
        # Update play count
        # This would typically update a database or analytics system
        logger.debug(f"Track {track_id} played by user {user_id} for {listen_duration} seconds")

async def handle_user_signup_events(events: List[StreamEvent]) -> None:
    """Handle user signup events."""
    for event in events:
        user_id = event.user_id
        signup_method = event.data.get('signup_method', 'unknown')
        
        # Initialize user analytics
        logger.info(f"New user signup: {user_id} via {signup_method}")

async def handle_subscription_events(events: List[StreamEvent]) -> None:
    """Handle subscription change events."""
    for event in events:
        user_id = event.user_id
        old_tier = event.data.get('old_tier', 'free')
        new_tier = event.data.get('new_tier', 'free')
        
        # Update user tier analytics
        logger.info(f"User {user_id} changed subscription from {old_tier} to {new_tier}")

if __name__ == "__main__":
    # Use uvloop for better performance
    uvloop.install()
    asyncio.run(main())
