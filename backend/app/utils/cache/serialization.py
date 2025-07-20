"""
Enterprise Cache Serialization
==============================
Advanced serialization and compression for optimal cache performance.

Expert Team Implementation:
- Lead Developer + AI Architect: Intelligent serialization selection and optimization
- Senior Backend Developer: High-performance async serialization with FastAPI integration
- Machine Learning Engineer: ML model serialization and tensor optimization
- DBA & Data Engineer: Schema-aware serialization and data format optimization
- Security Specialist: Encrypted serialization and secure data handling
- Microservices Architect: Cross-service serialization protocols and compatibility
"""

import asyncio
import base64
import gzip
import hashlib
import json
import logging
import pickle
import struct
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Type
import io

# External dependencies for advanced features
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    msgpack = None

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    lz4 = None

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    brotli = None

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None

logger = logging.getLogger(__name__)

# === Types and Enums ===
SerializedData = bytes
CompressionRatio = float

class SerializationFormat(Enum):
    """Serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    CUSTOM = "custom"

class CompressionAlgorithm(Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    BROTLI = "brotli"
    ZSTD = "zstd"

class EncryptionMethod(Enum):
    """Encryption methods."""
    NONE = "none"
    FERNET = "fernet"
    AES256 = "aes256"

@dataclass
class SerializationMetrics:
    """Serialization performance metrics."""
    serialization_time_ms: float = 0.0
    deserialization_time_ms: float = 0.0
    original_size_bytes: int = 0
    serialized_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 1.0
    format_used: SerializationFormat = SerializationFormat.JSON
    compression_used: CompressionAlgorithm = CompressionAlgorithm.NONE
    encryption_used: EncryptionMethod = EncryptionMethod.NONE
    
    @property
    def size_reduction_percent(self) -> float:
        """Calculate size reduction percentage."""
        if self.original_size_bytes == 0:
            return 0.0
        final_size = self.compressed_size_bytes or self.serialized_size_bytes
        reduction = (self.original_size_bytes - final_size) / self.original_size_bytes
        return reduction * 100

@dataclass
class SerializationConfig:
    """Configuration for serialization process."""
    format: SerializationFormat = SerializationFormat.JSON
    compression: CompressionAlgorithm = CompressionAlgorithm.LZ4
    compression_level: int = 6
    encryption: EncryptionMethod = EncryptionMethod.NONE
    encryption_key: Optional[bytes] = None
    auto_detect_format: bool = True
    optimize_for_size: bool = True
    optimize_for_speed: bool = False
    schema_validation: bool = False

# === Abstract Serializer Interface ===
class CacheSerializer(ABC):
    """Abstract base class for cache serializers."""
    
    def __init__(self, name: str, config: SerializationConfig = None):
        self.name = name
        self.config = config or SerializationConfig()
        self.metrics_history: List[SerializationMetrics] = []
        logger.info(f"Initialized serializer: {name}")
    
    @abstractmethod
    async def serialize(self, data: Any) -> Tuple[SerializedData, SerializationMetrics]:
        """Serialize data to bytes."""
        pass
    
    @abstractmethod
    async def deserialize(self, data: SerializedData) -> Tuple[Any, SerializationMetrics]:
        """Deserialize bytes to data."""
        pass
    
    def supports_type(self, data_type: Type) -> bool:
        """Check if serializer supports the given data type."""
        return True  # Default: support all types
    
    def get_average_metrics(self) -> SerializationMetrics:
        """Get average metrics from history."""
        if not self.metrics_history:
            return SerializationMetrics()
        
        avg_metrics = SerializationMetrics()
        count = len(self.metrics_history)
        
        for metrics in self.metrics_history:
            avg_metrics.serialization_time_ms += metrics.serialization_time_ms
            avg_metrics.deserialization_time_ms += metrics.deserialization_time_ms
            avg_metrics.original_size_bytes += metrics.original_size_bytes
            avg_metrics.serialized_size_bytes += metrics.serialized_size_bytes
            avg_metrics.compressed_size_bytes += metrics.compressed_size_bytes
            avg_metrics.compression_ratio += metrics.compression_ratio
        
        # Calculate averages
        avg_metrics.serialization_time_ms /= count
        avg_metrics.deserialization_time_ms /= count
        avg_metrics.original_size_bytes //= count
        avg_metrics.serialized_size_bytes //= count
        avg_metrics.compressed_size_bytes //= count
        avg_metrics.compression_ratio /= count
        
        return avg_metrics
    
    def _record_metrics(self, metrics: SerializationMetrics):
        """Record metrics for analysis."""
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics (last 1000)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

# === JSON Serializer ===
class JSONSerializer(CacheSerializer):
    """High-performance JSON serializer with compression."""
    
    def __init__(self, config: SerializationConfig = None):
        super().__init__("JSON", config)
        self.json_encoder = self._create_optimized_encoder()
    
    async def serialize(self, data: Any) -> Tuple[SerializedData, SerializationMetrics]:
        """Serialize data to JSON with compression."""
        start_time = time.time()
        metrics = SerializationMetrics(format_used=SerializationFormat.JSON)
        
        try:
            # Calculate original size (approximate)
            original_size = len(str(data).encode('utf-8'))
            metrics.original_size_bytes = original_size
            
            # Serialize to JSON
            if self.config.optimize_for_speed:
                json_data = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
            else:
                json_data = json.dumps(data, cls=self.json_encoder, ensure_ascii=False)
            
            json_bytes = json_data.encode('utf-8')
            metrics.serialized_size_bytes = len(json_bytes)
            
            # Apply compression
            final_data, compression_metrics = await self._apply_compression(json_bytes)
            metrics.compressed_size_bytes = len(final_data)
            metrics.compression_used = compression_metrics
            metrics.compression_ratio = original_size / len(final_data) if final_data else 1.0
            
            # Apply encryption if configured
            if self.config.encryption != EncryptionMethod.NONE:
                final_data = await self._apply_encryption(final_data)
                metrics.encryption_used = self.config.encryption
            
            # Record timing
            metrics.serialization_time_ms = (time.time() - start_time) * 1000
            self._record_metrics(metrics)
            
            return final_data, metrics
            
        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            raise
    
    async def deserialize(self, data: SerializedData) -> Tuple[Any, SerializationMetrics]:
        """Deserialize JSON with decompression."""
        start_time = time.time()
        metrics = SerializationMetrics(format_used=SerializationFormat.JSON)
        
        try:
            # Apply decryption if needed
            if self.config.encryption != EncryptionMethod.NONE:
                data = await self._apply_decryption(data)
            
            # Apply decompression
            json_bytes, compression_used = await self._apply_decompression(data)
            metrics.compression_used = compression_used
            
            # Deserialize from JSON
            json_str = json_bytes.decode('utf-8')
            result = json.loads(json_str)
            
            # Record timing
            metrics.deserialization_time_ms = (time.time() - start_time) * 1000
            
            return result, metrics
            
        except Exception as e:
            logger.error(f"JSON deserialization failed: {e}")
            raise
    
    def supports_type(self, data_type: Type) -> bool:
        """Check if type is JSON serializable."""
        json_types = (str, int, float, bool, list, dict, type(None))
        return issubclass(data_type, json_types)
    
    def _create_optimized_encoder(self):
        """Create optimized JSON encoder for common types."""
        class OptimizedJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                elif hasattr(obj, '_asdict'):  # namedtuple
                    return obj._asdict()
                return super().default(obj)
        
        return OptimizedJSONEncoder
    
    async def _apply_compression(self, data: bytes) -> Tuple[bytes, CompressionAlgorithm]:
        """Apply compression to data."""
        if self.config.compression == CompressionAlgorithm.NONE:
            return data, CompressionAlgorithm.NONE
        
        try:
            if self.config.compression == CompressionAlgorithm.GZIP:
                compressed = gzip.compress(data, compresslevel=self.config.compression_level)
                return compressed, CompressionAlgorithm.GZIP
            
            elif self.config.compression == CompressionAlgorithm.LZ4 and LZ4_AVAILABLE:
                compressed = lz4.frame.compress(data, compression_level=self.config.compression_level)
                return compressed, CompressionAlgorithm.LZ4
            
            elif self.config.compression == CompressionAlgorithm.BROTLI and BROTLI_AVAILABLE:
                compressed = brotli.compress(data, quality=self.config.compression_level)
                return compressed, CompressionAlgorithm.BROTLI
            
            else:
                # Fallback to gzip
                compressed = gzip.compress(data, compresslevel=self.config.compression_level)
                return compressed, CompressionAlgorithm.GZIP
                
        except Exception as e:
            logger.warning(f"Compression failed: {e}, using uncompressed data")
            return data, CompressionAlgorithm.NONE
    
    async def _apply_decompression(self, data: bytes) -> Tuple[bytes, CompressionAlgorithm]:
        """Apply decompression to data."""
        # Try to detect compression format by magic bytes
        if data.startswith(b'\x1f\x8b'):  # GZIP magic
            return gzip.decompress(data), CompressionAlgorithm.GZIP
        elif LZ4_AVAILABLE and data.startswith(b'\x04"M\x18'):  # LZ4 magic
            return lz4.frame.decompress(data), CompressionAlgorithm.LZ4
        elif BROTLI_AVAILABLE and self._is_brotli_data(data):
            return brotli.decompress(data), CompressionAlgorithm.BROTLI
        else:
            # Assume uncompressed
            return data, CompressionAlgorithm.NONE
    
    def _is_brotli_data(self, data: bytes) -> bool:
        """Check if data is Brotli compressed."""
        try:
            brotli.decompress(data)
            return True
        except:
            return False
    
    async def _apply_encryption(self, data: bytes) -> bytes:
        """Apply encryption to data."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Encryption requested but cryptography not available")
            return data
        
        if self.config.encryption == EncryptionMethod.FERNET:
            if not self.config.encryption_key:
                # Generate a key if not provided
                self.config.encryption_key = Fernet.generate_key()
            
            f = Fernet(self.config.encryption_key)
            return f.encrypt(data)
        
        return data
    
    async def _apply_decryption(self, data: bytes) -> bytes:
        """Apply decryption to data."""
        if not CRYPTO_AVAILABLE:
            return data
        
        if self.config.encryption == EncryptionMethod.FERNET:
            if not self.config.encryption_key:
                raise ValueError("Encryption key required for decryption")
            
            f = Fernet(self.config.encryption_key)
            return f.decrypt(data)
        
        return data

# === Pickle Serializer ===
class PickleSerializer(CacheSerializer):
    """High-performance Pickle serializer for Python objects."""
    
    def __init__(self, config: SerializationConfig = None):
        super().__init__("Pickle", config)
        self.pickle_protocol = pickle.HIGHEST_PROTOCOL
    
    async def serialize(self, data: Any) -> Tuple[SerializedData, SerializationMetrics]:
        """Serialize data using pickle."""
        start_time = time.time()
        metrics = SerializationMetrics(format_used=SerializationFormat.PICKLE)
        
        try:
            # Serialize with pickle
            pickled_data = pickle.dumps(data, protocol=self.pickle_protocol)
            metrics.original_size_bytes = len(pickled_data)
            metrics.serialized_size_bytes = len(pickled_data)
            
            # Apply compression
            final_data, compression_used = await self._apply_compression(pickled_data)
            metrics.compressed_size_bytes = len(final_data)
            metrics.compression_used = compression_used
            metrics.compression_ratio = len(pickled_data) / len(final_data) if final_data else 1.0
            
            # Apply encryption if configured
            if self.config.encryption != EncryptionMethod.NONE:
                final_data = await self._apply_encryption(final_data)
                metrics.encryption_used = self.config.encryption
            
            metrics.serialization_time_ms = (time.time() - start_time) * 1000
            self._record_metrics(metrics)
            
            return final_data, metrics
            
        except Exception as e:
            logger.error(f"Pickle serialization failed: {e}")
            raise
    
    async def deserialize(self, data: SerializedData) -> Tuple[Any, SerializationMetrics]:
        """Deserialize pickle data."""
        start_time = time.time()
        metrics = SerializationMetrics(format_used=SerializationFormat.PICKLE)
        
        try:
            # Apply decryption if needed
            if self.config.encryption != EncryptionMethod.NONE:
                data = await self._apply_decryption(data)
            
            # Apply decompression
            pickled_data, compression_used = await self._apply_decompression(data)
            metrics.compression_used = compression_used
            
            # Deserialize with pickle
            result = pickle.loads(pickled_data)
            
            metrics.deserialization_time_ms = (time.time() - start_time) * 1000
            
            return result, metrics
            
        except Exception as e:
            logger.error(f"Pickle deserialization failed: {e}")
            raise
    
    def supports_type(self, data_type: Type) -> bool:
        """Pickle supports almost all Python types."""
        return True
    
    async def _apply_compression(self, data: bytes) -> Tuple[bytes, CompressionAlgorithm]:
        """Apply compression optimized for binary data."""
        if self.config.compression == CompressionAlgorithm.NONE:
            return data, CompressionAlgorithm.NONE
        
        try:
            if self.config.compression == CompressionAlgorithm.LZ4 and LZ4_AVAILABLE:
                # LZ4 is very fast and good for binary data
                compressed = lz4.frame.compress(data, compression_level=self.config.compression_level)
                return compressed, CompressionAlgorithm.LZ4
            
            elif self.config.compression == CompressionAlgorithm.GZIP:
                compressed = gzip.compress(data, compresslevel=self.config.compression_level)
                return compressed, CompressionAlgorithm.GZIP
            
            else:
                # Default to LZ4 for binary data
                if LZ4_AVAILABLE:
                    compressed = lz4.frame.compress(data)
                    return compressed, CompressionAlgorithm.LZ4
                else:
                    compressed = gzip.compress(data)
                    return compressed, CompressionAlgorithm.GZIP
                
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return data, CompressionAlgorithm.NONE
    
    async def _apply_decompression(self, data: bytes) -> Tuple[bytes, CompressionAlgorithm]:
        """Apply decompression for binary data."""
        # Detect compression format
        if data.startswith(b'\x1f\x8b'):  # GZIP
            return gzip.decompress(data), CompressionAlgorithm.GZIP
        elif LZ4_AVAILABLE and data.startswith(b'\x04"M\x18'):  # LZ4
            return lz4.frame.decompress(data), CompressionAlgorithm.LZ4
        else:
            return data, CompressionAlgorithm.NONE
    
    async def _apply_encryption(self, data: bytes) -> bytes:
        """Apply encryption (same as JSON serializer)."""
        if not CRYPTO_AVAILABLE:
            return data
        
        if self.config.encryption == EncryptionMethod.FERNET:
            if not self.config.encryption_key:
                self.config.encryption_key = Fernet.generate_key()
            
            f = Fernet(self.config.encryption_key)
            return f.encrypt(data)
        
        return data
    
    async def _apply_decryption(self, data: bytes) -> bytes:
        """Apply decryption (same as JSON serializer)."""
        if not CRYPTO_AVAILABLE:
            return data
        
        if self.config.encryption == EncryptionMethod.FERNET:
            if not self.config.encryption_key:
                raise ValueError("Encryption key required for decryption")
            
            f = Fernet(self.config.encryption_key)
            return f.decrypt(data)
        
        return data

# === MessagePack Serializer ===
class MessagePackSerializer(CacheSerializer):
    """MessagePack serializer for efficient binary serialization."""
    
    def __init__(self, config: SerializationConfig = None):
        super().__init__("MessagePack", config)
        
        if not MSGPACK_AVAILABLE:
            raise ImportError("MessagePack not available. Install with: pip install msgpack")
    
    async def serialize(self, data: Any) -> Tuple[SerializedData, SerializationMetrics]:
        """Serialize data using MessagePack."""
        start_time = time.time()
        metrics = SerializationMetrics(format_used=SerializationFormat.MSGPACK)
        
        try:
            # Serialize with msgpack
            packed_data = msgpack.packb(data, use_bin_type=True)
            metrics.original_size_bytes = self._estimate_object_size(data)
            metrics.serialized_size_bytes = len(packed_data)
            
            # Apply compression
            final_data, compression_used = await self._apply_compression(packed_data)
            metrics.compressed_size_bytes = len(final_data)
            metrics.compression_used = compression_used
            metrics.compression_ratio = len(packed_data) / len(final_data) if final_data else 1.0
            
            metrics.serialization_time_ms = (time.time() - start_time) * 1000
            self._record_metrics(metrics)
            
            return final_data, metrics
            
        except Exception as e:
            logger.error(f"MessagePack serialization failed: {e}")
            raise
    
    async def deserialize(self, data: SerializedData) -> Tuple[Any, SerializationMetrics]:
        """Deserialize MessagePack data."""
        start_time = time.time()
        metrics = SerializationMetrics(format_used=SerializationFormat.MSGPACK)
        
        try:
            # Apply decompression
            packed_data, compression_used = await self._apply_decompression(data)
            metrics.compression_used = compression_used
            
            # Deserialize with msgpack
            result = msgpack.unpackb(packed_data, raw=False)
            
            metrics.deserialization_time_ms = (time.time() - start_time) * 1000
            
            return result, metrics
            
        except Exception as e:
            logger.error(f"MessagePack deserialization failed: {e}")
            raise
    
    def supports_type(self, data_type: Type) -> bool:
        """MessagePack supports most basic types."""
        supported_types = (str, int, float, bool, list, dict, tuple, bytes, type(None))
        return issubclass(data_type, supported_types)
    
    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate object size for metrics."""
        try:
            return len(pickle.dumps(obj))
        except:
            return len(str(obj).encode('utf-8'))
    
    async def _apply_compression(self, data: bytes) -> Tuple[bytes, CompressionAlgorithm]:
        """Apply compression optimized for MessagePack."""
        if self.config.compression == CompressionAlgorithm.NONE:
            return data, CompressionAlgorithm.NONE
        
        # LZ4 works very well with MessagePack
        if LZ4_AVAILABLE:
            compressed = lz4.frame.compress(data)
            return compressed, CompressionAlgorithm.LZ4
        else:
            compressed = gzip.compress(data)
            return compressed, CompressionAlgorithm.GZIP
    
    async def _apply_decompression(self, data: bytes) -> Tuple[bytes, CompressionAlgorithm]:
        """Apply decompression for MessagePack."""
        if data.startswith(b'\x1f\x8b'):  # GZIP
            return gzip.decompress(data), CompressionAlgorithm.GZIP
        elif LZ4_AVAILABLE and data.startswith(b'\x04"M\x18'):  # LZ4
            return lz4.frame.decompress(data), CompressionAlgorithm.LZ4
        else:
            return data, CompressionAlgorithm.NONE

# === Adaptive Serializer ===
class AdaptiveSerializer(CacheSerializer):
    """Adaptive serializer that chooses optimal format based on data type and size."""
    
    def __init__(self, config: SerializationConfig = None):
        super().__init__("Adaptive", config)
        
        # Initialize available serializers
        self.serializers = {
            SerializationFormat.JSON: JSONSerializer(config),
            SerializationFormat.PICKLE: PickleSerializer(config)
        }
        
        if MSGPACK_AVAILABLE:
            self.serializers[SerializationFormat.MSGPACK] = MessagePackSerializer(config)
        
        # Performance cache for format selection
        self.format_performance: Dict[Type, Dict[SerializationFormat, float]] = {}
    
    async def serialize(self, data: Any) -> Tuple[SerializedData, SerializationMetrics]:
        """Serialize using optimal format for data type."""
        data_type = type(data)
        optimal_format = await self._select_optimal_format(data, data_type)
        
        serializer = self.serializers[optimal_format]
        result, metrics = await serializer.serialize(data)
        
        # Update performance cache
        self._update_performance_cache(data_type, optimal_format, metrics)
        
        return result, metrics
    
    async def deserialize(self, data: SerializedData) -> Tuple[Any, SerializationMetrics]:
        """Deserialize by detecting format."""
        format_detected = await self._detect_format(data)
        
        if format_detected in self.serializers:
            serializer = self.serializers[format_detected]
            return await serializer.deserialize(data)
        else:
            # Try all serializers
            for serializer in self.serializers.values():
                try:
                    return await serializer.deserialize(data)
                except:
                    continue
            
            raise ValueError("Could not deserialize data with any available serializer")
    
    async def _select_optimal_format(self, data: Any, data_type: Type) -> SerializationFormat:
        """Select optimal serialization format."""
        # Check cache for previous performance
        if data_type in self.format_performance:
            perf_data = self.format_performance[data_type]
            if perf_data:
                # Select format with best performance (lowest time + size)
                best_format = min(perf_data.keys(), 
                                key=lambda fmt: perf_data[fmt])
                return best_format
        
        # Default selection based on data type and configuration
        if self.config.auto_detect_format:
            # Simple heuristics for format selection
            if isinstance(data, (dict, list)) and all(isinstance(k, str) for k in (data.keys() if isinstance(data, dict) else [])):
                # JSON-friendly data
                if MSGPACK_AVAILABLE and self.config.optimize_for_size:
                    return SerializationFormat.MSGPACK
                else:
                    return SerializationFormat.JSON
            else:
                # Complex Python objects
                return SerializationFormat.PICKLE
        else:
            return self.config.format
    
    async def _detect_format(self, data: SerializedData) -> Optional[SerializationFormat]:
        """Detect serialization format from data."""
        # Try to detect based on content patterns
        try:
            # Check for JSON (starts with { or [)
            if data.startswith((b'{', b'[')):
                return SerializationFormat.JSON
            
            # Check for pickle (has pickle opcodes)
            if data.startswith(b'\x80'):  # Pickle protocol markers
                return SerializationFormat.PICKLE
            
            # Check for MessagePack
            if MSGPACK_AVAILABLE and data[0:1] in [b'\x80', b'\x81', b'\x82', b'\x83', b'\x84', b'\x85', b'\x86', b'\x87', b'\x88', b'\x89', b'\x8a', b'\x8b', b'\x8c', b'\x8d', b'\x8e', b'\x8f']:
                return SerializationFormat.MSGPACK
            
        except (IndexError, TypeError):
            pass
        
        return None
    
    def _update_performance_cache(self, data_type: Type, format_used: SerializationFormat, metrics: SerializationMetrics):
        """Update performance cache with new metrics."""
        if data_type not in self.format_performance:
            self.format_performance[data_type] = {}
        
        # Calculate performance score (lower is better)
        score = (metrics.serialization_time_ms + 
                metrics.deserialization_time_ms + 
                metrics.compressed_size_bytes / 1000)  # Convert bytes to ms equivalent
        
        self.format_performance[data_type][format_used] = score

# === Factory Functions ===
def create_json_serializer(config: SerializationConfig = None) -> JSONSerializer:
    """Create JSON serializer with configuration."""
    return JSONSerializer(config)

def create_pickle_serializer(config: SerializationConfig = None) -> PickleSerializer:
    """Create Pickle serializer with configuration."""
    return PickleSerializer(config)

def create_msgpack_serializer(config: SerializationConfig = None) -> MessagePackSerializer:
    """Create MessagePack serializer with configuration."""
    return MessagePackSerializer(config)

def create_adaptive_serializer(config: SerializationConfig = None) -> AdaptiveSerializer:
    """Create adaptive serializer with configuration."""
    return AdaptiveSerializer(config)

def create_enterprise_serializer(optimize_for: str = "balanced") -> AdaptiveSerializer:
    """Create enterprise serializer optimized for specific use case."""
    config = SerializationConfig()
    
    if optimize_for == "speed":
        config.compression = CompressionAlgorithm.LZ4
        config.compression_level = 1
        config.optimize_for_speed = True
        config.auto_detect_format = True
    elif optimize_for == "size":
        config.compression = CompressionAlgorithm.BROTLI if BROTLI_AVAILABLE else CompressionAlgorithm.GZIP
        config.compression_level = 9
        config.optimize_for_size = True
        config.auto_detect_format = True
    elif optimize_for == "security":
        config.encryption = EncryptionMethod.FERNET if CRYPTO_AVAILABLE else EncryptionMethod.NONE
        config.compression = CompressionAlgorithm.LZ4
        config.auto_detect_format = True
    else:  # balanced
        config.compression = CompressionAlgorithm.LZ4
        config.compression_level = 6
        config.auto_detect_format = True
    
    return AdaptiveSerializer(config)
