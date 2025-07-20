"""
Enterprise Streaming Helpers
============================
Real-time streaming optimization utilities for Spotify AI Agent platform.

Expert Team Implementation:
- Lead Developer + AI Architect: Intelligent stream optimization and ML-driven QoS
- Senior Backend Developer: High-performance async streaming and buffer management
- Microservices Architect: Distributed streaming coordination and load balancing
- DBA & Data Engineer: Stream data processing and real-time analytics
- Security Specialist: Secure streaming protocols and DRM protection
- ML Engineer: Adaptive bitrate and intelligent stream prediction
"""

import asyncio
import logging
import json
import time
import threading
import queue
import struct
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Iterator, AsyncIterator
from abc import ABC, abstractmethod
from enum import Enum
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import weakref

# Networking and streaming imports
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

# Audio streaming imports
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

# Redis for stream coordination
try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    aioredis = None

logger = logging.getLogger(__name__)

# === Streaming Types and Enums ===
class StreamType(Enum):
    """Types of streams supported."""
    AUDIO = "audio"
    VIDEO = "video"
    METADATA = "metadata"
    CONTROL = "control"
    ANALYTICS = "analytics"

class StreamQuality(Enum):
    """Stream quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LOSSLESS = "lossless"
    ADAPTIVE = "adaptive"

class BufferState(Enum):
    """Buffer state indicators."""
    EMPTY = "empty"
    LOW = "low"
    OPTIMAL = "optimal"
    HIGH = "high"
    OVERFLOWING = "overflowing"

class StreamState(Enum):
    """Stream connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    BUFFERING = "buffering"
    PLAYING = "playing"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class StreamMetrics:
    """Real-time streaming metrics."""
    bitrate_kbps: float = 0.0
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    buffer_health: float = 1.0
    quality_score: float = 1.0
    throughput_mbps: float = 0.0
    connection_stability: float = 1.0
    
@dataclass
class QoSMetrics:
    """Quality of Service metrics."""
    mean_opinion_score: float = 5.0  # 1-5 scale
    perceptual_quality: float = 1.0  # 0-1 scale
    rebuffering_ratio: float = 0.0
    startup_time_ms: float = 0.0
    seek_time_ms: float = 0.0
    error_rate: float = 0.0

@dataclass
class StreamChunk:
    """Stream data chunk."""
    chunk_id: str
    stream_id: str
    sequence_number: int
    timestamp: datetime
    data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_level: StreamQuality = StreamQuality.MEDIUM
    is_key_frame: bool = False

@dataclass
class StreamStats:
    """Comprehensive stream statistics."""
    stream_id: str
    start_time: datetime
    duration_seconds: float
    bytes_transferred: int
    chunks_processed: int
    avg_bitrate_kbps: float
    peak_bitrate_kbps: float
    min_latency_ms: float
    max_latency_ms: float
    avg_latency_ms: float
    quality_switches: int
    rebuffering_events: int
    total_rebuffering_time_ms: float

# === Base Stream Processor ===
class BaseStreamProcessor(ABC):
    """Abstract base class for stream processors."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.stream_id = config.get('stream_id', f"stream_{int(time.time())}")
        self.state = StreamState.DISCONNECTED
        self.metrics = StreamMetrics()
        self.qos_metrics = QoSMetrics()
        self.stats = StreamStats(
            stream_id=self.stream_id,
            start_time=datetime.now(),
            duration_seconds=0.0,
            bytes_transferred=0,
            chunks_processed=0,
            avg_bitrate_kbps=0.0,
            peak_bitrate_kbps=0.0,
            min_latency_ms=float('inf'),
            max_latency_ms=0.0,
            avg_latency_ms=0.0,
            quality_switches=0,
            rebuffering_events=0,
            total_rebuffering_time_ms=0.0
        )
        self.callbacks = defaultdict(list)
        
    @abstractmethod
    async def start_stream(self, **kwargs):
        """Start stream processing."""
        pass
    
    @abstractmethod
    async def stop_stream(self):
        """Stop stream processing."""
        pass
    
    @abstractmethod
    async def process_chunk(self, chunk: StreamChunk) -> bool:
        """Process stream chunk."""
        pass
    
    def register_callback(self, event: str, callback: Callable):
        """Register event callback."""
        self.callbacks[event].append(callback)
    
    async def emit_event(self, event: str, data: Any = None):
        """Emit event to registered callbacks."""
        for callback in self.callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for event {event}: {e}")

# === Audio Stream Processor ===
class AudioStreamProcessor(BaseStreamProcessor):
    """High-performance audio stream processor."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.sample_rate = config.get('sample_rate', 44100)
        self.channels = config.get('channels', 2)
        self.bit_depth = config.get('bit_depth', 16)
        self.chunk_size = config.get('chunk_size', 1024)
        self.buffer_manager = None
        self.audio_stream = None
        
    async def start_stream(self, **kwargs):
        """Start audio stream processing."""
        try:
            self.state = StreamState.CONNECTING
            
            # Initialize buffer manager
            self.buffer_manager = BufferManager({
                'buffer_size': self.config.get('buffer_size', 8192),
                'low_water_mark': self.config.get('low_water_mark', 0.25),
                'high_water_mark': self.config.get('high_water_mark', 0.75)
            })
            
            # Start audio stream if PyAudio available
            if PYAUDIO_AVAILABLE and kwargs.get('enable_playback', False):
                await self._init_audio_playback()
            
            self.state = StreamState.CONNECTED
            self.stats.start_time = datetime.now()
            
            await self.emit_event('stream_started', {
                'stream_id': self.stream_id,
                'config': self.config
            })
            
            logger.info(f"Audio stream {self.stream_id} started")
            
        except Exception as e:
            self.state = StreamState.ERROR
            logger.error(f"Failed to start audio stream: {e}")
            raise
    
    async def stop_stream(self):
        """Stop audio stream processing."""
        try:
            self.state = StreamState.DISCONNECTED
            
            if self.audio_stream and PYAUDIO_AVAILABLE:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            
            if self.buffer_manager:
                await self.buffer_manager.clear()
            
            # Update final stats
            self.stats.duration_seconds = (datetime.now() - self.stats.start_time).total_seconds()
            
            await self.emit_event('stream_stopped', {
                'stream_id': self.stream_id,
                'stats': self.stats
            })
            
            logger.info(f"Audio stream {self.stream_id} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}")
    
    async def process_chunk(self, chunk: StreamChunk) -> bool:
        """Process audio chunk."""
        try:
            start_time = time.time()
            
            # Update stats
            self.stats.chunks_processed += 1
            self.stats.bytes_transferred += len(chunk.data)
            
            # Calculate latency
            latency_ms = (datetime.now() - chunk.timestamp).total_seconds() * 1000
            self.metrics.latency_ms = latency_ms
            
            # Update latency stats
            if latency_ms < self.stats.min_latency_ms:
                self.stats.min_latency_ms = latency_ms
            if latency_ms > self.stats.max_latency_ms:
                self.stats.max_latency_ms = latency_ms
            
            # Calculate average latency
            total_latency = self.stats.avg_latency_ms * (self.stats.chunks_processed - 1) + latency_ms
            self.stats.avg_latency_ms = total_latency / self.stats.chunks_processed
            
            # Buffer management
            if self.buffer_manager:
                await self.buffer_manager.add_chunk(chunk)
                
                # Check buffer state and adjust quality if needed
                buffer_state = self.buffer_manager.get_buffer_state()
                if buffer_state == BufferState.LOW:
                    await self._handle_low_buffer()
                elif buffer_state == BufferState.HIGH:
                    await self._handle_high_buffer()
            
            # Audio processing
            if chunk.stream_type == StreamType.AUDIO:
                await self._process_audio_data(chunk.data)
            
            # Update bitrate
            processing_time = time.time() - start_time
            chunk_bitrate = (len(chunk.data) * 8) / (processing_time * 1000)  # kbps
            self.metrics.bitrate_kbps = chunk_bitrate
            
            if chunk_bitrate > self.stats.peak_bitrate_kbps:
                self.stats.peak_bitrate_kbps = chunk_bitrate
            
            # Update average bitrate
            total_bitrate = self.stats.avg_bitrate_kbps * (self.stats.chunks_processed - 1) + chunk_bitrate
            self.stats.avg_bitrate_kbps = total_bitrate / self.stats.chunks_processed
            
            # Emit chunk processed event
            await self.emit_event('chunk_processed', {
                'chunk_id': chunk.chunk_id,
                'latency_ms': latency_ms,
                'bitrate_kbps': chunk_bitrate
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return False
    
    async def _init_audio_playback(self):
        """Initialize audio playback stream."""
        if not PYAUDIO_AVAILABLE:
            return
        
        try:
            p = pyaudio.PyAudio()
            
            self.audio_stream = p.open(
                format=p.get_format_from_width(self.bit_depth // 8),
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("Audio playback stream initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio playback: {e}")
    
    async def _process_audio_data(self, audio_data: bytes):
        """Process raw audio data."""
        try:
            # Play audio if stream available
            if self.audio_stream and PYAUDIO_AVAILABLE:
                self.audio_stream.write(audio_data)
            
            # Audio analysis and quality metrics
            await self._analyze_audio_quality(audio_data)
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
    
    async def _analyze_audio_quality(self, audio_data: bytes):
        """Analyze audio quality metrics."""
        try:
            # Convert bytes to numpy array for analysis
            if self.bit_depth == 16:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            elif self.bit_depth == 24:
                audio_array = np.frombuffer(audio_data, dtype=np.int32)  # Simplified
            else:
                return
            
            # Calculate RMS (energy)
            rms = np.sqrt(np.mean(audio_array.astype(np.float64)**2))
            
            # Calculate dynamic range
            dynamic_range = np.max(audio_array) - np.min(audio_array) if len(audio_array) > 0 else 0
            
            # Calculate zero crossing rate (indicator of frequency content)
            zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
            zcr = zero_crossings / len(audio_array) if len(audio_array) > 0 else 0
            
            # Update quality metrics
            self.qos_metrics.perceptual_quality = min(1.0, rms / 32767.0)  # Normalized for 16-bit
            
            # Simple quality score based on dynamic range and energy
            quality_score = min(1.0, (dynamic_range / 65535.0) * (rms / 16384.0))
            self.metrics.quality_score = quality_score
            
        except Exception as e:
            logger.error(f"Audio quality analysis error: {e}")
    
    async def _handle_low_buffer(self):
        """Handle low buffer condition."""
        self.qos_metrics.rebuffering_events += 1
        
        await self.emit_event('buffer_low', {
            'stream_id': self.stream_id,
            'buffer_level': self.buffer_manager.get_fill_percentage()
        })
        
        # Request lower quality to improve buffer
        await self.emit_event('quality_adjustment_request', {
            'stream_id': self.stream_id,
            'requested_quality': StreamQuality.LOW,
            'reason': 'buffer_underrun'
        })
    
    async def _handle_high_buffer(self):
        """Handle high buffer condition."""
        await self.emit_event('buffer_high', {
            'stream_id': self.stream_id,
            'buffer_level': self.buffer_manager.get_fill_percentage()
        })
        
        # Request higher quality if buffer allows
        await self.emit_event('quality_adjustment_request', {
            'stream_id': self.stream_id,
            'requested_quality': StreamQuality.HIGH,
            'reason': 'buffer_surplus'
        })

# === Buffer Manager ===
class BufferManager:
    """Intelligent buffer management for streaming."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.buffer_size = config.get('buffer_size', 8192)
        self.low_water_mark = config.get('low_water_mark', 0.25)
        self.high_water_mark = config.get('high_water_mark', 0.75)
        
        self.buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = asyncio.Lock()
        self.bytes_buffered = 0
        self.total_bytes_processed = 0
        self.buffer_history = deque(maxlen=100)
        
    async def add_chunk(self, chunk: StreamChunk):
        """Add chunk to buffer."""
        async with self.buffer_lock:
            if len(self.buffer) >= self.buffer_size:
                # Remove oldest chunk
                removed_chunk = self.buffer.popleft()
                self.bytes_buffered -= len(removed_chunk.data)
            
            self.buffer.append(chunk)
            self.bytes_buffered += len(chunk.data)
            self.total_bytes_processed += len(chunk.data)
            
            # Record buffer state
            self.buffer_history.append({
                'timestamp': datetime.now(),
                'fill_percentage': self.get_fill_percentage(),
                'bytes_buffered': self.bytes_buffered
            })
    
    async def get_chunk(self) -> Optional[StreamChunk]:
        """Get next chunk from buffer."""
        async with self.buffer_lock:
            if self.buffer:
                chunk = self.buffer.popleft()
                self.bytes_buffered -= len(chunk.data)
                return chunk
            return None
    
    def get_buffer_state(self) -> BufferState:
        """Get current buffer state."""
        fill_percentage = self.get_fill_percentage()
        
        if fill_percentage == 0:
            return BufferState.EMPTY
        elif fill_percentage < self.low_water_mark:
            return BufferState.LOW
        elif fill_percentage > self.high_water_mark:
            return BufferState.HIGH
        elif fill_percentage >= 0.95:
            return BufferState.OVERFLOWING
        else:
            return BufferState.OPTIMAL
    
    def get_fill_percentage(self) -> float:
        """Get buffer fill percentage."""
        return len(self.buffer) / self.buffer_size if self.buffer_size > 0 else 0.0
    
    def get_buffer_health(self) -> float:
        """Get buffer health score (0-1)."""
        fill_percentage = self.get_fill_percentage()
        
        # Optimal around 50%
        if fill_percentage < 0.5:
            return fill_percentage / 0.5
        else:
            return 1.0 - ((fill_percentage - 0.5) / 0.5)
    
    async def clear(self):
        """Clear buffer."""
        async with self.buffer_lock:
            self.buffer.clear()
            self.bytes_buffered = 0
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'current_size': len(self.buffer),
            'max_size': self.buffer_size,
            'fill_percentage': self.get_fill_percentage(),
            'bytes_buffered': self.bytes_buffered,
            'total_bytes_processed': self.total_bytes_processed,
            'buffer_health': self.get_buffer_health(),
            'state': self.get_buffer_state().value
        }

# === QoS Manager ===
class QoSManager:
    """Quality of Service management and optimization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.target_latency_ms = config.get('target_latency_ms', 100)
        self.min_buffer_ms = config.get('min_buffer_ms', 1000)
        self.max_jitter_ms = config.get('max_jitter_ms', 50)
        
        self.metrics_history = deque(maxlen=1000)
        self.quality_adjustments = deque(maxlen=100)
        self.current_quality = StreamQuality.MEDIUM
        
        # ML-based prediction
        self.latency_predictor = LatencyPredictor()
        self.bandwidth_estimator = BandwidthEstimator()
        
    async def update_metrics(self, metrics: StreamMetrics):
        """Update QoS metrics."""
        timestamp = datetime.now()
        
        # Store metrics history
        self.metrics_history.append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        # Calculate jitter
        if len(self.metrics_history) >= 2:
            prev_latency = self.metrics_history[-2]['metrics'].latency_ms
            current_latency = metrics.latency_ms
            jitter = abs(current_latency - prev_latency)
            metrics.jitter_ms = jitter
        
        # Update quality score
        quality_score = await self._calculate_quality_score(metrics)
        metrics.quality_score = quality_score
        
        # Check if quality adjustment is needed
        await self._check_quality_adjustment(metrics)
    
    async def _calculate_quality_score(self, metrics: StreamMetrics) -> float:
        """Calculate overall quality score."""
        # Weighted quality score based on multiple factors
        latency_score = max(0, 1 - (metrics.latency_ms / (self.target_latency_ms * 2)))
        jitter_score = max(0, 1 - (metrics.jitter_ms / self.max_jitter_ms))
        buffer_score = metrics.buffer_health
        throughput_score = min(1.0, metrics.throughput_mbps / 10.0)  # Assuming 10 Mbps is excellent
        
        # Weighted average
        weights = {'latency': 0.3, 'jitter': 0.2, 'buffer': 0.3, 'throughput': 0.2}
        quality_score = (
            latency_score * weights['latency'] +
            jitter_score * weights['jitter'] +
            buffer_score * weights['buffer'] +
            throughput_score * weights['throughput']
        )
        
        return max(0.0, min(1.0, quality_score))
    
    async def _check_quality_adjustment(self, metrics: StreamMetrics):
        """Check if quality adjustment is needed."""
        # Predict future performance
        predicted_latency = await self.latency_predictor.predict_latency(metrics)
        estimated_bandwidth = await self.bandwidth_estimator.estimate_bandwidth(metrics)
        
        # Determine optimal quality
        optimal_quality = await self._determine_optimal_quality(
            metrics, predicted_latency, estimated_bandwidth
        )
        
        # Adjust quality if needed
        if optimal_quality != self.current_quality:
            await self._adjust_quality(optimal_quality, metrics)
    
    async def _determine_optimal_quality(self, 
                                       metrics: StreamMetrics, 
                                       predicted_latency: float,
                                       estimated_bandwidth: float) -> StreamQuality:
        """Determine optimal quality level."""
        # Quality thresholds (simplified)
        quality_requirements = {
            StreamQuality.LOSSLESS: {'bandwidth_mbps': 5.0, 'max_latency_ms': 50},
            StreamQuality.HIGH: {'bandwidth_mbps': 2.0, 'max_latency_ms': 100},
            StreamQuality.MEDIUM: {'bandwidth_mbps': 1.0, 'max_latency_ms': 200},
            StreamQuality.LOW: {'bandwidth_mbps': 0.5, 'max_latency_ms': 500}
        }
        
        # Start with highest quality and work down
        for quality in [StreamQuality.LOSSLESS, StreamQuality.HIGH, StreamQuality.MEDIUM, StreamQuality.LOW]:
            req = quality_requirements[quality]
            
            if (estimated_bandwidth >= req['bandwidth_mbps'] and 
                predicted_latency <= req['max_latency_ms'] and
                metrics.buffer_health > 0.5):
                return quality
        
        return StreamQuality.LOW
    
    async def _adjust_quality(self, new_quality: StreamQuality, metrics: StreamMetrics):
        """Adjust stream quality."""
        old_quality = self.current_quality
        self.current_quality = new_quality
        
        # Record quality adjustment
        self.quality_adjustments.append({
            'timestamp': datetime.now(),
            'old_quality': old_quality.value,
            'new_quality': new_quality.value,
            'trigger_metrics': {
                'latency_ms': metrics.latency_ms,
                'buffer_health': metrics.buffer_health,
                'quality_score': metrics.quality_score
            }
        })
        
        logger.info(f"Quality adjusted from {old_quality.value} to {new_quality.value}")
    
    def get_qos_report(self) -> Dict[str, Any]:
        """Generate QoS report."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = [item['metrics'] for item in self.metrics_history[-100:]]
        
        return {
            'current_quality': self.current_quality.value,
            'avg_latency_ms': statistics.mean([m.latency_ms for m in recent_metrics]),
            'avg_jitter_ms': statistics.mean([m.jitter_ms for m in recent_metrics]),
            'avg_quality_score': statistics.mean([m.quality_score for m in recent_metrics]),
            'avg_buffer_health': statistics.mean([m.buffer_health for m in recent_metrics]),
            'quality_adjustments_count': len(self.quality_adjustments),
            'stability_score': self._calculate_stability_score()
        }
    
    def _calculate_stability_score(self) -> float:
        """Calculate connection stability score."""
        if len(self.quality_adjustments) < 2:
            return 1.0
        
        # Fewer quality changes = more stable
        recent_adjustments = len([adj for adj in self.quality_adjustments 
                                if (datetime.now() - adj['timestamp']).total_seconds() < 300])
        
        # Normalize to 0-1 scale (fewer adjustments = higher score)
        stability_score = max(0.0, 1.0 - (recent_adjustments / 10.0))
        return stability_score

# === Latency Optimizer ===
class LatencyOptimizer:
    """Advanced latency optimization for real-time streaming."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.optimization_strategies = []
        self.latency_history = deque(maxlen=1000)
        self.optimization_results = deque(maxlen=100)
        
        # Initialize optimization strategies
        self._init_optimization_strategies()
    
    def _init_optimization_strategies(self):
        """Initialize latency optimization strategies."""
        self.optimization_strategies = [
            {'name': 'buffer_sizing', 'func': self._optimize_buffer_size},
            {'name': 'quality_adaptation', 'func': self._optimize_quality_adaptation},
            {'name': 'network_prediction', 'func': self._optimize_network_prediction},
            {'name': 'chunk_prioritization', 'func': self._optimize_chunk_priority}
        ]
    
    async def optimize_latency(self, stream_processor: BaseStreamProcessor) -> Dict[str, Any]:
        """Optimize latency for stream processor."""
        optimization_results = {}
        
        for strategy in self.optimization_strategies:
            try:
                result = await strategy['func'](stream_processor)
                optimization_results[strategy['name']] = result
            except Exception as e:
                logger.error(f"Optimization strategy {strategy['name']} failed: {e}")
                optimization_results[strategy['name']] = {'success': False, 'error': str(e)}
        
        # Record optimization results
        self.optimization_results.append({
            'timestamp': datetime.now(),
            'results': optimization_results,
            'latency_before': stream_processor.metrics.latency_ms
        })
        
        return optimization_results
    
    async def _optimize_buffer_size(self, stream_processor: BaseStreamProcessor) -> Dict[str, Any]:
        """Optimize buffer size based on network conditions."""
        if not hasattr(stream_processor, 'buffer_manager') or not stream_processor.buffer_manager:
            return {'success': False, 'reason': 'No buffer manager available'}
        
        buffer_manager = stream_processor.buffer_manager
        current_latency = stream_processor.metrics.latency_ms
        
        # Analyze buffer performance
        buffer_stats = buffer_manager.get_buffer_stats()
        
        # Adaptive buffer sizing
        if current_latency > 200 and buffer_stats['fill_percentage'] < 0.3:
            # Increase buffer size for high latency
            new_size = int(buffer_manager.buffer_size * 1.2)
            buffer_manager.buffer_size = min(new_size, 16384)  # Cap at 16K
            
            return {
                'success': True,
                'action': 'increased_buffer',
                'old_size': buffer_manager.buffer_size,
                'new_size': new_size
            }
        
        elif current_latency < 50 and buffer_stats['fill_percentage'] > 0.8:
            # Decrease buffer size for low latency
            new_size = int(buffer_manager.buffer_size * 0.8)
            buffer_manager.buffer_size = max(new_size, 1024)  # Minimum 1K
            
            return {
                'success': True,
                'action': 'decreased_buffer',
                'old_size': buffer_manager.buffer_size,
                'new_size': new_size
            }
        
        return {'success': True, 'action': 'no_change', 'reason': 'buffer_size_optimal'}
    
    async def _optimize_quality_adaptation(self, stream_processor: BaseStreamProcessor) -> Dict[str, Any]:
        """Optimize quality adaptation strategy."""
        current_latency = stream_processor.metrics.latency_ms
        quality_score = stream_processor.metrics.quality_score
        
        # Predictive quality adjustment
        if current_latency > 150 and quality_score > 0.7:
            # Proactively reduce quality to prevent buffering
            return {
                'success': True,
                'recommendation': 'reduce_quality',
                'reason': 'latency_prevention',
                'confidence': 0.8
            }
        
        elif current_latency < 50 and quality_score < 0.9:
            # Increase quality when conditions allow
            return {
                'success': True,
                'recommendation': 'increase_quality',
                'reason': 'performance_headroom',
                'confidence': 0.7
            }
        
        return {'success': True, 'recommendation': 'maintain_quality', 'reason': 'conditions_stable'}
    
    async def _optimize_network_prediction(self, stream_processor: BaseStreamProcessor) -> Dict[str, Any]:
        """Optimize using network condition prediction."""
        # Simple network prediction based on recent history
        recent_latencies = [item['metrics'].latency_ms for item in stream_processor.metrics_history[-10:]]
        
        if len(recent_latencies) < 3:
            return {'success': False, 'reason': 'insufficient_history'}
        
        # Calculate trend
        latency_trend = statistics.mean(recent_latencies[-3:]) - statistics.mean(recent_latencies[:3])
        
        if latency_trend > 20:  # Increasing latency
            return {
                'success': True,
                'prediction': 'degrading_network',
                'recommended_action': 'preemptive_quality_reduction',
                'confidence': min(0.9, abs(latency_trend) / 100)
            }
        
        elif latency_trend < -20:  # Improving latency
            return {
                'success': True,
                'prediction': 'improving_network',
                'recommended_action': 'quality_upgrade_opportunity',
                'confidence': min(0.9, abs(latency_trend) / 100)
            }
        
        return {'success': True, 'prediction': 'stable_network', 'recommended_action': 'maintain_current'}
    
    async def _optimize_chunk_priority(self, stream_processor: BaseStreamProcessor) -> Dict[str, Any]:
        """Optimize chunk processing priority."""
        # Implement intelligent chunk prioritization
        # Key frames and audio chunks get higher priority
        
        return {
            'success': True,
            'strategy': 'key_frame_priority',
            'description': 'Prioritize key frames and audio chunks for smoother playback'
        }

# === ML-based Predictors ===
class LatencyPredictor:
    """Machine learning based latency prediction."""
    
    def __init__(self):
        self.history = deque(maxlen=100)
        self.model_trained = False
    
    async def predict_latency(self, current_metrics: StreamMetrics) -> float:
        """Predict future latency based on current metrics."""
        # Add current metrics to history
        self.history.append({
            'timestamp': datetime.now(),
            'latency': current_metrics.latency_ms,
            'jitter': current_metrics.jitter_ms,
            'buffer_health': current_metrics.buffer_health,
            'throughput': current_metrics.throughput_mbps
        })
        
        # Simple prediction using moving average and trend
        if len(self.history) < 5:
            return current_metrics.latency_ms
        
        recent_latencies = [item['latency'] for item in list(self.history)[-5:]]
        
        # Linear trend prediction
        x = list(range(len(recent_latencies)))
        y = recent_latencies
        
        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Predict next value
            predicted_latency = slope * n + intercept
            
            # Bound the prediction to reasonable values
            return max(0, min(predicted_latency, current_metrics.latency_ms * 2))
        
        return current_metrics.latency_ms

class BandwidthEstimator:
    """Bandwidth estimation for adaptive streaming."""
    
    def __init__(self):
        self.bandwidth_history = deque(maxlen=50)
        
    async def estimate_bandwidth(self, current_metrics: StreamMetrics) -> float:
        """Estimate available bandwidth."""
        # Record current throughput
        self.bandwidth_history.append({
            'timestamp': datetime.now(),
            'throughput_mbps': current_metrics.throughput_mbps
        })
        
        if len(self.bandwidth_history) < 3:
            return current_metrics.throughput_mbps
        
        # Calculate weighted average (recent measurements have higher weight)
        weights = [i + 1 for i in range(len(self.bandwidth_history))]
        weighted_sum = sum(item['throughput_mbps'] * weights[i] 
                          for i, item in enumerate(self.bandwidth_history))
        weight_sum = sum(weights)
        
        estimated_bandwidth = weighted_sum / weight_sum if weight_sum > 0 else current_metrics.throughput_mbps
        
        return estimated_bandwidth

# === Stream Coordinator ===
class StreamCoordinator:
    """Coordinate multiple streams and load balancing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.active_streams = {}
        self.stream_stats = defaultdict(dict)
        self.load_balancer = LoadBalancer(config)
        
        # Redis for distributed coordination
        if REDIS_AVAILABLE:
            self.redis_client = None
            self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection for distributed coordination."""
        try:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url)
            logger.info("Redis client initialized for stream coordination")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            self.redis_client = None
    
    async def register_stream(self, stream_processor: BaseStreamProcessor):
        """Register stream with coordinator."""
        stream_id = stream_processor.stream_id
        self.active_streams[stream_id] = stream_processor
        
        # Store in Redis for distributed coordination
        if self.redis_client:
            try:
                stream_info = {
                    'stream_id': stream_id,
                    'node_id': self.config.get('node_id', 'default'),
                    'start_time': datetime.now().isoformat(),
                    'stream_type': stream_processor.config.get('stream_type', 'audio')
                }
                
                await self.redis_client.setex(
                    f"stream:{stream_id}",
                    3600,  # TTL 1 hour
                    json.dumps(stream_info)
                )
            except Exception as e:
                logger.error(f"Failed to register stream in Redis: {e}")
        
        logger.info(f"Stream {stream_id} registered with coordinator")
    
    async def unregister_stream(self, stream_id: str):
        """Unregister stream from coordinator."""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
        
        # Remove from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(f"stream:{stream_id}")
            except Exception as e:
                logger.error(f"Failed to unregister stream from Redis: {e}")
        
        logger.info(f"Stream {stream_id} unregistered from coordinator")
    
    async def get_stream_load(self) -> Dict[str, Any]:
        """Get current stream load metrics."""
        total_streams = len(self.active_streams)
        total_bandwidth = sum(
            stream.metrics.throughput_mbps 
            for stream in self.active_streams.values()
        )
        avg_latency = statistics.mean([
            stream.metrics.latency_ms 
            for stream in self.active_streams.values()
        ]) if self.active_streams else 0
        
        return {
            'total_streams': total_streams,
            'total_bandwidth_mbps': total_bandwidth,
            'avg_latency_ms': avg_latency,
            'node_capacity_utilization': total_streams / self.config.get('max_streams', 100),
            'bandwidth_utilization': total_bandwidth / self.config.get('max_bandwidth_mbps', 1000)
        }
    
    async def optimize_stream_distribution(self):
        """Optimize stream distribution across nodes."""
        load_metrics = await self.get_stream_load()
        
        # If load is high, request load balancing
        if (load_metrics['node_capacity_utilization'] > 0.8 or 
            load_metrics['bandwidth_utilization'] > 0.8):
            
            await self.load_balancer.rebalance_streams(self.active_streams)

class LoadBalancer:
    """Load balancing for distributed streaming."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.node_metrics = defaultdict(dict)
    
    async def rebalance_streams(self, active_streams: Dict[str, BaseStreamProcessor]):
        """Rebalance streams across available nodes."""
        # Simplified load balancing logic
        # In production, implement more sophisticated algorithms
        
        logger.info(f"Rebalancing {len(active_streams)} streams")
        
        # For now, just log the action
        # Real implementation would migrate streams to less loaded nodes
        return True

# === Factory Functions ===
def create_audio_stream_processor(config: Dict[str, Any] = None) -> AudioStreamProcessor:
    """Create audio stream processor."""
    return AudioStreamProcessor(config)

def create_buffer_manager(config: Dict[str, Any] = None) -> BufferManager:
    """Create buffer manager."""
    return BufferManager(config)

def create_qos_manager(config: Dict[str, Any] = None) -> QoSManager:
    """Create QoS manager."""
    return QoSManager(config)

def create_latency_optimizer(config: Dict[str, Any] = None) -> LatencyOptimizer:
    """Create latency optimizer."""
    return LatencyOptimizer(config)

def create_stream_coordinator(config: Dict[str, Any] = None) -> StreamCoordinator:
    """Create stream coordinator."""
    return StreamCoordinator(config)

# === Export Classes ===
__all__ = [
    'BaseStreamProcessor', 'AudioStreamProcessor', 'BufferManager', 
    'QoSManager', 'LatencyOptimizer', 'StreamCoordinator', 'LoadBalancer',
    'StreamType', 'StreamQuality', 'BufferState', 'StreamState',
    'StreamMetrics', 'QoSMetrics', 'StreamChunk', 'StreamStats',
    'LatencyPredictor', 'BandwidthEstimator',
    'create_audio_stream_processor', 'create_buffer_manager', 
    'create_qos_manager', 'create_latency_optimizer', 'create_stream_coordinator'
]
