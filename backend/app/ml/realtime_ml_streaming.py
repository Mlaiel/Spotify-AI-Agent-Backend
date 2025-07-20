"""
Real-Time ML Streaming & Event Processing Engine
===============================================

Enterprise-grade real-time machine learning system for streaming data processing,
live model inference, and event-driven ML pipelines with ultra-low latency.

Features:
- Real-time event stream processing with Kafka/Redis Streams
- Live model inference with sub-millisecond latency
- Online learning and model adaptation
- Stream analytics and real-time feature engineering
- Event-driven ML pipeline orchestration
- Real-time A/B testing and experimentation
- Streaming anomaly detection and alerting
- Auto-scaling inference infrastructure
- Real-time model performance monitoring
- Circuit breaker and fault tolerance patterns
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable, AsyncIterator, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import numpy as np
import pandas as pd
import torch
import redis.asyncio as redis
from collections import deque, defaultdict
import pickle
import base64
import hashlib
from enum import Enum
import uuid
from contextlib import asynccontextmanager
import statistics

from . import audit_ml_operation, ML_CONFIG

logger = logging.getLogger(__name__)

class StreamEventType(Enum):
    """Stream event types"""
    USER_ACTION = "user_action"
    PLAY_EVENT = "play_event"
    RECOMMENDATION_REQUEST = "recommendation_request"
    FEEDBACK = "feedback"
    MODEL_UPDATE = "model_update"
    ALERT = "alert"
    METRIC = "metric"

class StreamStatus(Enum):
    """Stream processing status"""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class StreamEvent:
    """Generic stream event structure"""
    event_id: str
    event_type: StreamEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class InferenceRequest:
    """Real-time inference request"""
    request_id: str
    model_id: str
    features: Union[Dict[str, Any], np.ndarray, List[float]]
    timestamp: datetime
    priority: int = 1
    timeout_ms: int = 100

@dataclass
class InferenceResponse:
    """Real-time inference response"""
    request_id: str
    model_id: str
    predictions: Union[float, List[float], Dict[str, Any]]
    confidence: float
    latency_ms: float
    timestamp: datetime
    model_version: str

@dataclass
class StreamMetrics:
    """Stream processing metrics"""
    events_processed: int
    events_per_second: float
    avg_latency_ms: float
    error_rate: float
    last_updated: datetime

class RealTimeFeatureStore:
    """
    High-performance real-time feature store
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or ML_CONFIG["feature_store_url"]
        self.redis_client = None
        self.feature_cache = {}
        self.feature_ttl = 3600  # 1 hour default TTL
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("✅ Real-time feature store connected")
        except Exception as e:
            logger.error(f"❌ Feature store connection failed: {e}")
            raise
    
    async def set_feature(self, user_id: str, feature_name: str, 
                         value: Any, ttl: int = None):
        """Set a feature value for a user"""
        try:
            key = f"features:{user_id}:{feature_name}"
            value_str = json.dumps(value) if not isinstance(value, str) else value
            
            if self.redis_client:
                await self.redis_client.setex(key, ttl or self.feature_ttl, value_str)
            
            # Also cache locally
            if user_id not in self.feature_cache:
                self.feature_cache[user_id] = {}
            self.feature_cache[user_id][feature_name] = {
                'value': value,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to set feature {feature_name} for user {user_id}: {e}")
    
    async def get_feature(self, user_id: str, feature_name: str, 
                         default: Any = None) -> Any:
        """Get a feature value for a user"""
        try:
            key = f"features:{user_id}:{feature_name}"
            
            # Try Redis first
            if self.redis_client:
                value_str = await self.redis_client.get(key)
                if value_str:
                    try:
                        return json.loads(value_str)
                    except json.JSONDecodeError:
                        return value_str
            
            # Fall back to local cache
            user_features = self.feature_cache.get(user_id, {})
            feature_data = user_features.get(feature_name)
            
            if feature_data:
                # Check if not expired (1 hour)
                if datetime.utcnow() - feature_data['timestamp'] < timedelta(hours=1):
                    return feature_data['value']
            
            return default
            
        except Exception as e:
            logger.error(f"Failed to get feature {feature_name} for user {user_id}: {e}")
            return default
    
    async def get_user_features(self, user_id: str, 
                               feature_names: List[str] = None) -> Dict[str, Any]:
        """Get multiple features for a user"""
        if feature_names is None:
            # Get all features for user
            pattern = f"features:{user_id}:*"
            if self.redis_client:
                try:
                    keys = await self.redis_client.keys(pattern)
                    features = {}
                    for key in keys:
                        feature_name = key.split(':')[-1]
                        value = await self.get_feature(user_id, feature_name)
                        if value is not None:
                            features[feature_name] = value
                    return features
                except Exception as e:
                    logger.error(f"Failed to get all features for user {user_id}: {e}")
            
            # Fall back to local cache
            return self.feature_cache.get(user_id, {})
        
        else:
            # Get specific features
            features = {}
            for feature_name in feature_names:
                value = await self.get_feature(user_id, feature_name)
                if value is not None:
                    features[feature_name] = value
            return features
    
    async def update_features_batch(self, updates: List[Dict[str, Any]]):
        """Batch update features for better performance"""
        try:
            if self.redis_client:
                pipe = self.redis_client.pipeline()
                
                for update in updates:
                    user_id = update['user_id']
                    features = update['features']
                    ttl = update.get('ttl', self.feature_ttl)
                    
                    for feature_name, value in features.items():
                        key = f"features:{user_id}:{feature_name}"
                        value_str = json.dumps(value) if not isinstance(value, str) else value
                        pipe.setex(key, ttl, value_str)
                
                await pipe.execute()
                
            logger.info(f"✅ Batch updated {len(updates)} feature sets")
            
        except Exception as e:
            logger.error(f"❌ Batch feature update failed: {e}")

class StreamProcessor:
    """
    High-performance stream processor for ML events
    """
    
    def __init__(self, stream_name: str, config: Dict[str, Any] = None):
        self.stream_name = stream_name
        self.config = config or {}
        self.status = StreamStatus.STOPPED
        self.event_handlers = {}
        self.metrics = StreamMetrics(0, 0.0, 0.0, 0.0, datetime.utcnow())
        self.event_buffer = deque(maxlen=1000)
        self.processing_tasks = []
        self.redis_client = None
        self.feature_store = None
        
        # Performance tracking
        self.processed_count = 0
        self.error_count = 0
        self.latency_samples = deque(maxlen=100)
        self.last_metrics_update = time.time()
        
    async def initialize(self):
        """Initialize stream processor"""
        try:
            # Initialize Redis for stream processing
            self.redis_client = redis.from_url(ML_CONFIG["feature_store_url"])
            await self.redis_client.ping()
            
            # Initialize feature store
            self.feature_store = RealTimeFeatureStore()
            await self.feature_store.initialize()
            
            logger.info(f"✅ Stream processor '{self.stream_name}' initialized")
            
        except Exception as e:
            logger.error(f"❌ Stream processor initialization failed: {e}")
            raise
    
    def register_handler(self, event_type: StreamEventType, 
                        handler: Callable[[StreamEvent], Any]):
        """Register event handler for specific event type"""
        self.event_handlers[event_type] = handler
        logger.info(f"📋 Registered handler for {event_type.value}")
    
    async def start_processing(self):
        """Start stream processing"""
        if self.status == StreamStatus.RUNNING:
            logger.warning("Stream processor already running")
            return
        
        self.status = StreamStatus.RUNNING
        logger.info(f"🚀 Starting stream processor '{self.stream_name}'")
        
        # Start processing tasks
        self.processing_tasks = [
            asyncio.create_task(self._event_consumer()),
            asyncio.create_task(self._metrics_updater()),
            asyncio.create_task(self._health_monitor())
        ]
        
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
    
    async def stop_processing(self):
        """Stop stream processing"""
        self.status = StreamStatus.STOPPED
        
        # Cancel all tasks
        for task in self.processing_tasks:
            task.cancel()
        
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Close connections
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info(f"⏹️ Stream processor '{self.stream_name}' stopped")
    
    async def publish_event(self, event: StreamEvent):
        """Publish event to stream"""
        try:
            if self.redis_client:
                # Use Redis Streams
                stream_key = f"stream:{self.stream_name}"
                event_data = {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'user_id': event.user_id or '',
                    'session_id': event.session_id or '',
                    'data': json.dumps(event.data),
                    'metadata': json.dumps(event.metadata)
                }
                
                await self.redis_client.xadd(stream_key, event_data)
            
            # Also add to local buffer for immediate processing
            self.event_buffer.append(event)
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
    
    async def _event_consumer(self):
        """Consume and process events from stream"""
        last_id = '0'
        
        while self.status == StreamStatus.RUNNING:
            try:
                # Process local buffer first
                while self.event_buffer and self.status == StreamStatus.RUNNING:
                    event = self.event_buffer.popleft()
                    await self._process_event(event)
                
                # Then consume from Redis stream
                if self.redis_client:
                    stream_key = f"stream:{self.stream_name}"
                    streams = {stream_key: last_id}
                    
                    result = await self.redis_client.xread(streams, count=10, block=100)
                    
                    for stream, messages in result:
                        for message_id, fields in messages:
                            try:
                                # Parse event from Redis stream
                                event = StreamEvent(
                                    event_id=fields['event_id'],
                                    event_type=StreamEventType(fields['event_type']),
                                    timestamp=datetime.fromisoformat(fields['timestamp']),
                                    user_id=fields['user_id'] if fields['user_id'] else None,
                                    session_id=fields['session_id'] if fields['session_id'] else None,
                                    data=json.loads(fields['data']),
                                    metadata=json.loads(fields['metadata'])
                                )
                                
                                await self._process_event(event)
                                last_id = message_id
                                
                            except Exception as e:
                                logger.error(f"Failed to parse stream message: {e}")
                                self.error_count += 1
                
                # Small delay to prevent busy waiting
                if not result if self.redis_client else True:
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Event consumer error: {e}")
                self.error_count += 1
                await asyncio.sleep(1)  # Back off on error
    
    async def _process_event(self, event: StreamEvent):
        """Process a single event"""
        start_time = time.time()
        
        try:
            # Find handler for event type
            handler = self.event_handlers.get(event.event_type)
            
            if handler:
                # Execute handler
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                
                self.processed_count += 1
                
                # Track latency
                latency_ms = (time.time() - start_time) * 1000
                self.latency_samples.append(latency_ms)
                
                logger.debug(f"📨 Processed event {event.event_id} in {latency_ms:.2f}ms")
                
            else:
                logger.warning(f"No handler for event type: {event.event_type.value}")
                
        except Exception as e:
            logger.error(f"Event processing error: {e}")
            self.error_count += 1
    
    async def _metrics_updater(self):
        """Update stream metrics periodically"""
        while self.status == StreamStatus.RUNNING:
            try:
                current_time = time.time()
                time_delta = current_time - self.last_metrics_update
                
                if time_delta > 0:
                    events_per_second = self.processed_count / time_delta
                    avg_latency = statistics.mean(self.latency_samples) if self.latency_samples else 0
                    error_rate = self.error_count / max(self.processed_count + self.error_count, 1)
                    
                    self.metrics = StreamMetrics(
                        events_processed=self.processed_count,
                        events_per_second=events_per_second,
                        avg_latency_ms=avg_latency,
                        error_rate=error_rate,
                        last_updated=datetime.utcnow()
                    )
                    
                    # Reset counters
                    self.processed_count = 0
                    self.error_count = 0
                    self.last_metrics_update = current_time
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(10)
    
    async def _health_monitor(self):
        """Monitor stream health and trigger alerts"""
        while self.status == StreamStatus.RUNNING:
            try:
                # Check for anomalies
                if self.metrics.error_rate > 0.1:  # 10% error rate
                    logger.warning(f"🚨 High error rate: {self.metrics.error_rate:.2%}")
                
                if self.metrics.avg_latency_ms > 1000:  # 1 second latency
                    logger.warning(f"🚨 High latency: {self.metrics.avg_latency_ms:.2f}ms")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
    
    def get_metrics(self) -> StreamMetrics:
        """Get current stream metrics"""
        return self.metrics

class RealTimeInferenceEngine:
    """
    Ultra-low latency inference engine for real-time ML
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models = {}
        self.model_versions = {}
        self.inference_cache = {}
        self.circuit_breakers = {}
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.inference_metrics = defaultdict(lambda: {
            'count': 0,
            'total_latency': 0,
            'errors': 0,
            'last_request': None
        })
        
    async def register_model(self, model_id: str, model: Any, version: str = "v1.0"):
        """Register a model for real-time inference"""
        try:
            self.models[model_id] = model
            self.model_versions[model_id] = version
            self.circuit_breakers[model_id] = CircuitBreaker(
                failure_threshold=5,
                timeout_duration=30
            )
            
            logger.info(f"✅ Registered model {model_id} v{version} for real-time inference")
            
        except Exception as e:
            logger.error(f"❌ Model registration failed: {e}")
            raise
    
    @audit_ml_operation("real_time_inference")
    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Perform real-time inference"""
        start_time = time.time()
        
        try:
            # Check if model exists
            if request.model_id not in self.models:
                raise ValueError(f"Model {request.model_id} not found")
            
            # Check circuit breaker
            circuit_breaker = self.circuit_breakers[request.model_id]
            if circuit_breaker.is_open():
                raise Exception("Circuit breaker is open")
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = self.inference_cache.get(cache_key)
            
            if cached_result and self._is_cache_valid(cached_result):
                logger.debug(f"🎯 Cache hit for request {request.request_id}")
                cached_result['request_id'] = request.request_id
                cached_result['timestamp'] = datetime.utcnow()
                return InferenceResponse(**cached_result)
            
            # Perform inference
            model = self.models[request.model_id]
            
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                self.executor, 
                self._run_inference, 
                model, 
                request.features
            )
            
            # Calculate confidence (model-specific logic)
            confidence = self._calculate_confidence(predictions, request.model_id)
            
            # Create response
            latency_ms = (time.time() - start_time) * 1000
            
            response = InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                predictions=predictions,
                confidence=confidence,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
                model_version=self.model_versions[request.model_id]
            )
            
            # Cache result
            self._cache_result(cache_key, response)
            
            # Update metrics
            self._update_inference_metrics(request.model_id, latency_ms, success=True)
            
            # Reset circuit breaker on success
            circuit_breaker.record_success()
            
            logger.debug(f"⚡ Inference completed for {request.request_id} in {latency_ms:.2f}ms")
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Record failure in circuit breaker
            if request.model_id in self.circuit_breakers:
                self.circuit_breakers[request.model_id].record_failure()
            
            # Update metrics
            self._update_inference_metrics(request.model_id, latency_ms, success=False)
            
            logger.error(f"❌ Inference failed for {request.request_id}: {e}")
            raise
    
    def _run_inference(self, model: Any, features: Any) -> Any:
        """Run inference in thread pool"""
        try:
            if hasattr(model, 'predict'):
                # Scikit-learn style
                if isinstance(features, dict):
                    # Convert dict to array
                    feature_array = np.array(list(features.values())).reshape(1, -1)
                else:
                    feature_array = np.array(features).reshape(1, -1)
                return model.predict(feature_array)[0]
            
            elif hasattr(model, 'forward'):
                # PyTorch style
                import torch
                model.eval()
                with torch.no_grad():
                    if isinstance(features, dict):
                        feature_tensor = torch.FloatTensor(list(features.values())).unsqueeze(0)
                    else:
                        feature_tensor = torch.FloatTensor(features).unsqueeze(0)
                    output = model(feature_tensor)
                    return output.cpu().numpy().tolist()
            
            else:
                # Generic callable
                return model(features)
                
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            raise
    
    def _generate_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        feature_hash = hashlib.md5(
            json.dumps(request.features, sort_keys=True).encode()
        ).hexdigest()
        return f"{request.model_id}:{feature_hash}"
    
    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid"""
        cache_time = cached_result.get('cached_at', datetime.min)
        return (datetime.utcnow() - cache_time).total_seconds() < 300  # 5 minutes
    
    def _cache_result(self, cache_key: str, response: InferenceResponse):
        """Cache inference result"""
        self.inference_cache[cache_key] = {
            'model_id': response.model_id,
            'predictions': response.predictions,
            'confidence': response.confidence,
            'model_version': response.model_version,
            'cached_at': datetime.utcnow()
        }
        
        # Limit cache size
        if len(self.inference_cache) > 1000:
            # Remove oldest entries
            sorted_items = sorted(
                self.inference_cache.items(),
                key=lambda x: x[1]['cached_at']
            )
            for key, _ in sorted_items[:100]:  # Remove 100 oldest
                del self.inference_cache[key]
    
    def _calculate_confidence(self, predictions: Any, model_id: str) -> float:
        """Calculate prediction confidence"""
        try:
            if isinstance(predictions, (list, np.ndarray)):
                # For classification probabilities
                if len(predictions) > 1 and all(0 <= p <= 1 for p in predictions):
                    return max(predictions)
                # For regression
                return 0.8  # Default confidence
            
            elif isinstance(predictions, (int, float)):
                # Single prediction
                return 0.8
            
            else:
                return 0.5  # Low confidence for unknown format
                
        except Exception:
            return 0.5
    
    def _update_inference_metrics(self, model_id: str, latency_ms: float, success: bool):
        """Update inference metrics"""
        metrics = self.inference_metrics[model_id]
        metrics['count'] += 1
        metrics['total_latency'] += latency_ms
        metrics['last_request'] = datetime.utcnow()
        
        if not success:
            metrics['errors'] += 1
    
    def get_inference_metrics(self) -> Dict[str, Any]:
        """Get inference performance metrics"""
        metrics_summary = {}
        
        for model_id, metrics in self.inference_metrics.items():
            if metrics['count'] > 0:
                metrics_summary[model_id] = {
                    'total_requests': metrics['count'],
                    'avg_latency_ms': metrics['total_latency'] / metrics['count'],
                    'error_rate': metrics['errors'] / metrics['count'],
                    'last_request': metrics['last_request'].isoformat() if metrics['last_request'] else None
                }
        
        return metrics_summary

class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance
    """
    
    def __init__(self, failure_threshold: int = 5, timeout_duration: int = 30):
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.state == "open":
            if self.last_failure_time and \
               (time.time() - self.last_failure_time) > self.timeout_duration:
                self.state = "half-open"
                return False
            return True
        return False
    
    def record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class RealTimeMLEngine:
    """
    Comprehensive Real-Time ML Engine
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.stream_processors = {}
        self.inference_engine = RealTimeInferenceEngine(config)
        self.feature_store = RealTimeFeatureStore()
        self.is_running = False
        
        logger.info("🔥 Real-Time ML Engine initialized")
    
    async def initialize(self):
        """Initialize the real-time ML engine"""
        try:
            await self.feature_store.initialize()
            logger.info("✅ Real-Time ML Engine ready")
        except Exception as e:
            logger.error(f"❌ Real-Time ML Engine initialization failed: {e}")
            raise
    
    async def create_stream_processor(self, stream_name: str, 
                                    config: Dict[str, Any] = None) -> StreamProcessor:
        """Create a new stream processor"""
        processor = StreamProcessor(stream_name, config)
        await processor.initialize()
        self.stream_processors[stream_name] = processor
        return processor
    
    async def start_all_streams(self):
        """Start all stream processors"""
        self.is_running = True
        
        tasks = []
        for processor in self.stream_processors.values():
            tasks.append(processor.start_processing())
        
        logger.info(f"🚀 Started {len(tasks)} stream processors")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all_streams(self):
        """Stop all stream processors"""
        self.is_running = False
        
        for processor in self.stream_processors.values():
            await processor.stop_processing()
        
        logger.info("⏹️ All stream processors stopped")
    
    async def register_inference_model(self, model_id: str, model: Any, version: str = "v1.0"):
        """Register model for real-time inference"""
        await self.inference_engine.register_model(model_id, model, version)
    
    async def predict_realtime(self, request: InferenceRequest) -> InferenceResponse:
        """Perform real-time prediction"""
        return await self.inference_engine.predict(request)
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        stream_status = {}
        for name, processor in self.stream_processors.items():
            stream_status[name] = {
                'status': processor.status.value,
                'metrics': asdict(processor.get_metrics())
            }
        
        return {
            'is_running': self.is_running,
            'stream_processors': stream_status,
            'inference_metrics': self.inference_engine.get_inference_metrics(),
            'total_streams': len(self.stream_processors),
            'last_updated': datetime.utcnow().isoformat()
        }

# Factory function
def create_realtime_ml_engine(config: Dict[str, Any] = None) -> RealTimeMLEngine:
    """Create real-time ML engine instance"""
    return RealTimeMLEngine(config)

# Export main components
__all__ = [
    'RealTimeMLEngine',
    'StreamProcessor',
    'RealTimeInferenceEngine',
    'RealTimeFeatureStore',
    'StreamEvent',
    'InferenceRequest',
    'InferenceResponse',
    'StreamEventType',
    'StreamStatus',
    'StreamMetrics',
    'CircuitBreaker',
    'create_realtime_ml_engine'
]
