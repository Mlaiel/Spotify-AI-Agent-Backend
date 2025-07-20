"""
Tests for cache serialization and data handling in Spotify AI Agent

Comprehensive testing suite for cache serialization, deserialization,
data compression, encoding and complex object handling.

Developed by Expert Team:
- Lead Dev + AI Architect
- Senior Backend Developer (Python/FastAPI/Django)
- Machine Learning Engineer (TensorFlow/PyTorch/Hugging Face)
- DBA & Data Engineer (PostgreSQL/Redis/MongoDB)
- Backend Security Specialist
- Microservices Architect
"""

import pytest
import json
import pickle
import gzip
import zlib
import bz2
import lzma
import base64
import hashlib
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from decimal import Decimal
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

from app.utils.cache.serialization import (
    CacheSerializer, CacheDeserializer, SerializationManager,
    JsonSerializer, PickleSerializer, MessagePackSerializer,
    CompressionHandler, EncodingHandler, ObjectConverter,
    CustomTypeHandler, SerializationMetrics
)
from app.utils.cache.exceptions import SerializationError, DeserializationError


@dataclass
class MockSpotifyTrack:
    """Mock Spotify track for testing"""
    id: str
    name: str
    artists: List[str]
    duration_ms: int
    popularity: int
    audio_features: Dict[str, float]
    release_date: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockMLModel:
    """Mock ML model for testing"""
    model_id: str
    model_type: str
    parameters: Dict[str, Any]
    weights: np.ndarray
    training_history: List[Dict[str, float]]
    created_at: datetime
    
    def predict(self, data):
        return np.random.random(len(data))


class TestCacheSerializer:
    """Test cache serialization"""
    
    @pytest.fixture
    def json_serializer(self):
        """JSON serializer fixture"""
        return JsonSerializer(
            ensure_ascii=False,
            indent=None,
            sort_keys=True,
            handle_datetime=True,
            handle_decimal=True
        )
    
    @pytest.fixture
    def pickle_serializer(self):
        """Pickle serializer fixture"""
        return PickleSerializer(
            protocol=pickle.HIGHEST_PROTOCOL,
            fix_imports=True
        )
    
    @pytest.fixture
    def msgpack_serializer(self):
        """MessagePack serializer fixture"""
        if not MSGPACK_AVAILABLE:
            pytest.skip("msgpack not available")
        return MessagePackSerializer(
            use_bin_type=True,
            strict_types=False
        )
    
    def test_json_serialization_basic(self, json_serializer):
        """Test basic JSON serialization"""
        serializer = json_serializer
        
        # Test basic types
        test_data = {
            "string": "hello world",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        serialized = serializer.serialize(test_data)
        assert isinstance(serialized, (str, bytes))
        
        # Should be valid JSON
        parsed = json.loads(serialized)
        assert parsed == test_data
    
    def test_json_serialization_datetime(self, json_serializer):
        """Test JSON serialization with datetime objects"""
        serializer = json_serializer
        
        now = datetime.now()
        test_data = {
            "timestamp": now,
            "date_list": [now, now + timedelta(hours=1)]
        }
        
        serialized = serializer.serialize(test_data)
        parsed = json.loads(serialized)
        
        # Datetime should be serialized as ISO format string
        assert isinstance(parsed["timestamp"], str)
        assert isinstance(parsed["date_list"][0], str)
    
    def test_json_serialization_decimal(self, json_serializer):
        """Test JSON serialization with Decimal objects"""
        serializer = json_serializer
        
        test_data = {
            "price": Decimal("19.99"),
            "prices": [Decimal("10.50"), Decimal("25.75")]
        }
        
        serialized = serializer.serialize(test_data)
        parsed = json.loads(serialized)
        
        # Decimal should be serialized as float or string
        assert isinstance(parsed["price"], (float, str))
        assert isinstance(parsed["prices"][0], (float, str))
    
    def test_json_serialization_complex_objects(self, json_serializer):
        """Test JSON serialization with complex objects"""
        serializer = json_serializer
        
        track = MockSpotifyTrack(
            id="track_123",
            name="Test Song",
            artists=["Artist 1", "Artist 2"],
            duration_ms=180000,
            popularity=85,
            audio_features={"energy": 0.8, "valence": 0.6},
            release_date=datetime.now()
        )
        
        # Should handle dataclass serialization
        serialized = serializer.serialize(track)
        assert isinstance(serialized, (str, bytes))
        
        # Should be parseable
        parsed = json.loads(serialized)
        assert parsed["id"] == "track_123"
        assert parsed["name"] == "Test Song"
    
    def test_pickle_serialization_basic(self, pickle_serializer):
        """Test basic pickle serialization"""
        serializer = pickle_serializer
        
        # Test various types including complex objects
        test_data = {
            "function": lambda x: x * 2,
            "complex": complex(1, 2),
            "set": {1, 2, 3},
            "tuple": (1, 2, 3),
            "nested": {"deep": {"very": {"nested": "value"}}}
        }
        
        serialized = serializer.serialize(test_data)
        assert isinstance(serialized, bytes)
        
        # Should be valid pickle data
        deserialized = pickle.loads(serialized)
        assert deserialized["complex"] == complex(1, 2)
        assert deserialized["set"] == {1, 2, 3}
        assert deserialized["function"](5) == 10
    
    def test_pickle_serialization_numpy(self, pickle_serializer):
        """Test pickle serialization with NumPy arrays"""
        serializer = pickle_serializer
        
        # Create numpy arrays
        test_data = {
            "array_1d": np.array([1, 2, 3, 4, 5]),
            "array_2d": np.array([[1, 2], [3, 4]]),
            "array_float": np.array([1.1, 2.2, 3.3]),
            "array_bool": np.array([True, False, True])
        }
        
        serialized = serializer.serialize(test_data)
        deserialized = pickle.loads(serialized)
        
        # Check numpy arrays are preserved
        np.testing.assert_array_equal(deserialized["array_1d"], test_data["array_1d"])
        np.testing.assert_array_equal(deserialized["array_2d"], test_data["array_2d"])
        np.testing.assert_array_almost_equal(deserialized["array_float"], test_data["array_float"])
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pickle_serialization_pandas(self, pickle_serializer):
        """Test pickle serialization with pandas objects"""
        serializer = pickle_serializer
        
        # Create pandas objects
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["a", "b", "c"],
            "C": [1.1, 2.2, 3.3]
        })
        
        series = pd.Series([1, 2, 3, 4, 5], name="test_series")
        
        test_data = {
            "dataframe": df,
            "series": series
        }
        
        serialized = serializer.serialize(test_data)
        deserialized = pickle.loads(serialized)
        
        # Check pandas objects are preserved
        pd.testing.assert_frame_equal(deserialized["dataframe"], df)
        pd.testing.assert_series_equal(deserialized["series"], series)
    
    @pytest.mark.skipif(not MSGPACK_AVAILABLE, reason="msgpack not available")
    def test_msgpack_serialization(self, msgpack_serializer):
        """Test MessagePack serialization"""
        serializer = msgpack_serializer
        
        test_data = {
            "string": "hello world",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "binary": b"binary data"
        }
        
        serialized = serializer.serialize(test_data)
        assert isinstance(serialized, bytes)
        
        # Should be valid msgpack data
        deserialized = msgpack.unpackb(serialized, raw=False)
        
        # Check basic equality (msgpack preserves types better than JSON)
        assert deserialized["string"] == test_data["string"]
        assert deserialized["integer"] == test_data["integer"]
        assert deserialized["binary"] == test_data["binary"]
    
    def test_serialization_error_handling(self, json_serializer):
        """Test serialization error handling"""
        serializer = json_serializer
        
        # Create non-serializable object
        class NonSerializable:
            def __init__(self):
                self.file = open(__file__, 'r')
        
        test_data = {
            "good": "data",
            "bad": NonSerializable()
        }
        
        # Should raise SerializationError
        with pytest.raises(SerializationError):
            serializer.serialize(test_data)
    
    def test_serialization_metrics(self, json_serializer):
        """Test serialization metrics collection"""
        serializer = json_serializer
        
        # Enable metrics
        serializer.enable_metrics(True)
        
        test_data = {"key": "value" * 1000}  # 5KB data
        
        serialized = serializer.serialize(test_data)
        metrics = serializer.get_metrics()
        
        assert "serialization_time" in metrics
        assert "input_size_bytes" in metrics
        assert "output_size_bytes" in metrics
        assert "compression_ratio" in metrics
        assert metrics["input_size_bytes"] > 0
        assert metrics["output_size_bytes"] > 0


class TestCacheDeserializer:
    """Test cache deserialization"""
    
    @pytest.fixture
    def json_deserializer(self):
        """JSON deserializer fixture"""
        return CacheDeserializer(format="json")
    
    @pytest.fixture
    def pickle_deserializer(self):
        """Pickle deserializer fixture"""
        return CacheDeserializer(format="pickle")
    
    def test_json_deserialization(self, json_serializer, json_deserializer):
        """Test JSON deserialization"""
        serializer = json_serializer
        deserializer = json_deserializer
        
        original_data = {
            "string": "hello",
            "number": 42,
            "nested": {"list": [1, 2, 3]}
        }
        
        # Serialize then deserialize
        serialized = serializer.serialize(original_data)
        deserialized = deserializer.deserialize(serialized)
        
        assert deserialized == original_data
    
    def test_pickle_deserialization(self, pickle_serializer, pickle_deserializer):
        """Test pickle deserialization"""
        serializer = pickle_serializer
        deserializer = pickle_deserializer
        
        original_data = {
            "set": {1, 2, 3},
            "tuple": (1, 2, 3),
            "complex": complex(1, 2)
        }
        
        # Serialize then deserialize
        serialized = serializer.serialize(original_data)
        deserialized = deserializer.deserialize(serialized)
        
        assert deserialized["set"] == original_data["set"]
        assert deserialized["tuple"] == original_data["tuple"]
        assert deserialized["complex"] == original_data["complex"]
    
    def test_deserialization_type_validation(self, json_deserializer):
        """Test deserialization with type validation"""
        deserializer = json_deserializer
        
        # Enable type validation
        deserializer.enable_type_validation(True)
        deserializer.set_expected_type(dict)
        
        # Valid data
        valid_data = json.dumps({"key": "value"})
        result = deserializer.deserialize(valid_data)
        assert isinstance(result, dict)
        
        # Invalid data (wrong type)
        invalid_data = json.dumps([1, 2, 3])  # List instead of dict
        
        with pytest.raises(DeserializationError):
            deserializer.deserialize(invalid_data)
    
    def test_deserialization_schema_validation(self, json_deserializer):
        """Test deserialization with schema validation"""
        deserializer = json_deserializer
        
        # Define schema
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "age": {"type": "number", "minimum": 0}
            },
            "required": ["id", "name"]
        }
        
        deserializer.set_schema(schema)
        
        # Valid data
        valid_data = json.dumps({
            "id": "123",
            "name": "John Doe",
            "age": 30
        })
        
        result = deserializer.deserialize(valid_data)
        assert result["id"] == "123"
        
        # Invalid data (missing required field)
        invalid_data = json.dumps({
            "id": "123"
            # Missing 'name'
        })
        
        with pytest.raises(DeserializationError):
            deserializer.deserialize(invalid_data)
    
    def test_deserialization_error_handling(self, json_deserializer):
        """Test deserialization error handling"""
        deserializer = json_deserializer
        
        # Invalid JSON
        invalid_json = '{"invalid": json}'
        
        with pytest.raises(DeserializationError):
            deserializer.deserialize(invalid_json)
        
        # Empty data
        with pytest.raises(DeserializationError):
            deserializer.deserialize("")
        
        # None data
        with pytest.raises(DeserializationError):
            deserializer.deserialize(None)
    
    def test_deserialization_with_custom_types(self, json_deserializer):
        """Test deserialization with custom type handlers"""
        deserializer = json_deserializer
        
        # Register custom type handler for datetime
        def datetime_handler(value):
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        
        deserializer.register_custom_handler("datetime", datetime_handler)
        
        # Data with datetime strings
        data_with_datetime = json.dumps({
            "timestamp": "2024-01-15T10:30:00Z",
            "__type_hints__": {"timestamp": "datetime"}
        })
        
        result = deserializer.deserialize(data_with_datetime)
        assert isinstance(result["timestamp"], datetime)


class TestCompressionHandler:
    """Test compression handling"""
    
    @pytest.fixture
    def compression_handler(self):
        """Compression handler fixture"""
        return CompressionHandler(
            default_algorithm="gzip",
            compression_threshold=100,  # Compress data > 100 bytes
            compression_level=6
        )
    
    def test_gzip_compression(self, compression_handler):
        """Test gzip compression"""
        handler = compression_handler
        
        # Large data that should be compressed
        data = "This is a test string that will be compressed. " * 100
        data_bytes = data.encode('utf-8')
        
        # Compress
        compressed = handler.compress(data_bytes, algorithm="gzip")
        assert len(compressed) < len(data_bytes)
        
        # Decompress
        decompressed = handler.decompress(compressed, algorithm="gzip")
        assert decompressed == data_bytes
    
    def test_zlib_compression(self, compression_handler):
        """Test zlib compression"""
        handler = compression_handler
        
        data = b"x" * 1000  # 1KB of data
        
        compressed = handler.compress(data, algorithm="zlib")
        assert len(compressed) < len(data)
        
        decompressed = handler.decompress(compressed, algorithm="zlib")
        assert decompressed == data
    
    def test_bz2_compression(self, compression_handler):
        """Test bz2 compression"""
        handler = compression_handler
        
        data = "repetitive data " * 200
        data_bytes = data.encode('utf-8')
        
        compressed = handler.compress(data_bytes, algorithm="bz2")
        assert len(compressed) < len(data_bytes)
        
        decompressed = handler.decompress(compressed, algorithm="bz2")
        assert decompressed == data_bytes
    
    def test_lzma_compression(self, compression_handler):
        """Test LZMA compression"""
        handler = compression_handler
        
        data = json.dumps({"key": "value"} * 1000).encode('utf-8')
        
        compressed = handler.compress(data, algorithm="lzma")
        assert len(compressed) < len(data)
        
        decompressed = handler.decompress(compressed, algorithm="lzma")
        assert decompressed == data
    
    def test_compression_threshold(self, compression_handler):
        """Test compression threshold"""
        handler = compression_handler
        
        # Small data (below threshold)
        small_data = b"small"
        result = handler.compress_if_beneficial(small_data)
        assert result == small_data  # Should not be compressed
        
        # Large data (above threshold)
        large_data = b"x" * 1000
        result = handler.compress_if_beneficial(large_data)
        assert len(result) < len(large_data)  # Should be compressed
    
    def test_compression_ratio_analysis(self, compression_handler):
        """Test compression ratio analysis"""
        handler = compression_handler
        
        # Different types of data
        test_cases = [
            ("random", b"".join(np.random.bytes(1000))),  # Random data (low compression)
            ("repetitive", b"a" * 1000),  # Repetitive data (high compression)
            ("json", json.dumps({"key": "value"} * 100).encode()),  # Structured data
            ("text", ("This is a sample text. " * 50).encode())  # Natural text
        ]
        
        for name, data in test_cases:
            ratios = handler.analyze_compression_ratios(data)
            
            assert "gzip" in ratios
            assert "zlib" in ratios
            assert "bz2" in ratios
            assert "lzma" in ratios
            
            # All ratios should be positive
            assert all(ratio > 0 for ratio in ratios.values())
    
    def test_best_compression_algorithm(self, compression_handler):
        """Test best compression algorithm selection"""
        handler = compression_handler
        
        # Highly repetitive data (should favor algorithms good at repetition)
        repetitive_data = ("abc" * 1000).encode()
        best_algo = handler.find_best_algorithm(repetitive_data)
        
        assert best_algo in ["gzip", "zlib", "bz2", "lzma"]
        
        # Verify it actually gives good compression
        compressed = handler.compress(repetitive_data, algorithm=best_algo)
        ratio = len(compressed) / len(repetitive_data)
        assert ratio < 0.1  # Should compress to less than 10% of original


class TestEncodingHandler:
    """Test encoding handling"""
    
    @pytest.fixture
    def encoding_handler(self):
        """Encoding handler fixture"""
        return EncodingHandler(
            default_encoding="utf-8",
            enable_base64=True,
            enable_hex=True
        )
    
    def test_base64_encoding(self, encoding_handler):
        """Test base64 encoding"""
        handler = encoding_handler
        
        # Binary data
        data = b"\x00\x01\x02\x03\xff\xfe\xfd"
        
        # Encode
        encoded = handler.base64_encode(data)
        assert isinstance(encoded, str)
        
        # Decode
        decoded = handler.base64_decode(encoded)
        assert decoded == data
    
    def test_hex_encoding(self, encoding_handler):
        """Test hex encoding"""
        handler = encoding_handler
        
        data = b"hello world"
        
        # Encode
        encoded = handler.hex_encode(data)
        assert isinstance(encoded, str)
        assert all(c in "0123456789abcdef" for c in encoded.lower())
        
        # Decode
        decoded = handler.hex_decode(encoded)
        assert decoded == data
    
    def test_url_safe_encoding(self, encoding_handler):
        """Test URL-safe encoding"""
        handler = encoding_handler
        
        # Data that might contain URL-unsafe characters when base64 encoded
        data = b"???><<<&&&+++"
        
        # URL-safe base64 encoding
        encoded = handler.url_safe_base64_encode(data)
        assert "+" not in encoded
        assert "/" not in encoded
        
        # Decode
        decoded = handler.url_safe_base64_decode(encoded)
        assert decoded == data
    
    def test_text_encoding(self, encoding_handler):
        """Test text encoding with different charsets"""
        handler = encoding_handler
        
        # Unicode text
        text = "Hello, 世界! 🌍"
        
        # Test different encodings
        for encoding in ["utf-8", "utf-16", "utf-32"]:
            encoded = handler.encode_text(text, encoding)
            assert isinstance(encoded, bytes)
            
            decoded = handler.decode_text(encoded, encoding)
            assert decoded == text
    
    def test_encoding_detection(self, encoding_handler):
        """Test automatic encoding detection"""
        handler = encoding_handler
        
        # Text in different encodings
        text = "Hello, world!"
        
        # Encode in different formats
        utf8_data = text.encode('utf-8')
        latin1_data = text.encode('latin-1')
        
        # Detect encodings
        utf8_detected = handler.detect_encoding(utf8_data)
        latin1_detected = handler.detect_encoding(latin1_data)
        
        assert utf8_detected in ["utf-8", "ascii"]  # ASCII is subset of UTF-8
        assert latin1_detected in ["latin-1", "ascii"]


class TestObjectConverter:
    """Test object conversion"""
    
    @pytest.fixture
    def object_converter(self):
        """Object converter fixture"""
        return ObjectConverter(
            enable_dataclass_support=True,
            enable_numpy_support=True,
            enable_pandas_support=PANDAS_AVAILABLE
        )
    
    def test_dataclass_conversion(self, object_converter):
        """Test dataclass conversion"""
        converter = object_converter
        
        track = MockSpotifyTrack(
            id="track_123",
            name="Test Song",
            artists=["Artist 1"],
            duration_ms=180000,
            popularity=85,
            audio_features={"energy": 0.8},
            release_date=datetime.now()
        )
        
        # Convert to dict
        dict_data = converter.to_dict(track)
        assert isinstance(dict_data, dict)
        assert dict_data["id"] == "track_123"
        assert dict_data["name"] == "Test Song"
        
        # Convert back to object
        restored_track = converter.from_dict(dict_data, MockSpotifyTrack)
        assert restored_track.id == track.id
        assert restored_track.name == track.name
    
    def test_numpy_conversion(self, object_converter):
        """Test NumPy array conversion"""
        converter = object_converter
        
        # Different numpy array types
        arrays = {
            "int_array": np.array([1, 2, 3, 4, 5]),
            "float_array": np.array([1.1, 2.2, 3.3]),
            "bool_array": np.array([True, False, True]),
            "2d_array": np.array([[1, 2], [3, 4]]),
            "string_array": np.array(["a", "b", "c"])
        }
        
        for name, array in arrays.items():
            # Convert to serializable format
            converted = converter.numpy_to_serializable(array)
            assert isinstance(converted, dict)
            assert "dtype" in converted
            assert "shape" in converted
            assert "data" in converted
            
            # Convert back to numpy
            restored = converter.serializable_to_numpy(converted)
            np.testing.assert_array_equal(restored, array)
    
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pandas_conversion(self, object_converter):
        """Test pandas objects conversion"""
        converter = object_converter
        
        # DataFrame
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
            "C": [1.1, 2.2, 3.3]
        })
        
        # Convert DataFrame
        df_converted = converter.pandas_to_serializable(df)
        assert isinstance(df_converted, dict)
        assert "type" in df_converted
        assert df_converted["type"] == "DataFrame"
        
        # Restore DataFrame
        df_restored = converter.serializable_to_pandas(df_converted)
        pd.testing.assert_frame_equal(df_restored, df)
        
        # Series
        series = pd.Series([1, 2, 3, 4], name="test_series")
        
        series_converted = converter.pandas_to_serializable(series)
        series_restored = converter.serializable_to_pandas(series_converted)
        pd.testing.assert_series_equal(series_restored, series)
    
    def test_datetime_conversion(self, object_converter):
        """Test datetime conversion"""
        converter = object_converter
        
        now = datetime.now()
        
        # Convert datetime to string
        dt_string = converter.datetime_to_string(now)
        assert isinstance(dt_string, str)
        
        # Convert back to datetime
        dt_restored = converter.string_to_datetime(dt_string)
        assert isinstance(dt_restored, datetime)
        
        # Should be equal (within microsecond precision)
        assert abs((dt_restored - now).total_seconds()) < 0.001
    
    def test_complex_object_conversion(self, object_converter):
        """Test complex nested object conversion"""
        converter = object_converter
        
        # Complex nested structure
        complex_data = {
            "track": MockSpotifyTrack(
                id="track_123",
                name="Test Song",
                artists=["Artist 1"],
                duration_ms=180000,
                popularity=85,
                audio_features={"energy": 0.8},
                release_date=datetime.now()
            ),
            "audio_data": np.array([0.1, 0.2, 0.3, 0.4]),
            "timestamps": [datetime.now(), datetime.now() + timedelta(minutes=1)],
            "metadata": {
                "nested": {
                    "values": [1, 2, 3]
                }
            }
        }
        
        # Convert entire structure
        converted = converter.convert_complex_object(complex_data)
        
        # Should be serializable (test by converting to JSON)
        json_str = json.dumps(converted, default=str)
        assert isinstance(json_str, str)
        
        # Restore structure
        restored = converter.restore_complex_object(converted)
        
        # Verify restoration
        assert restored["track"].id == complex_data["track"].id
        assert restored["metadata"]["nested"]["values"] == [1, 2, 3]


class TestCustomTypeHandler:
    """Test custom type handling"""
    
    @pytest.fixture
    def type_handler(self):
        """Custom type handler fixture"""
        return CustomTypeHandler()
    
    def test_register_custom_serializer(self, type_handler):
        """Test registering custom serializers"""
        handler = type_handler
        
        # Custom class
        class CustomClass:
            def __init__(self, value):
                self.value = value
        
        # Register custom serializer
        def custom_serializer(obj):
            return {"__custom_type__": "CustomClass", "value": obj.value}
        
        def custom_deserializer(data):
            return CustomClass(data["value"])
        
        handler.register_type(CustomClass, custom_serializer, custom_deserializer)
        
        # Test serialization
        obj = CustomClass("test_value")
        serialized = handler.serialize(obj)
        
        assert serialized["__custom_type__"] == "CustomClass"
        assert serialized["value"] == "test_value"
        
        # Test deserialization
        deserialized = handler.deserialize(serialized)
        assert isinstance(deserialized, CustomClass)
        assert deserialized.value == "test_value"
    
    def test_ml_model_serialization(self, type_handler):
        """Test ML model serialization"""
        handler = type_handler
        
        # Register ML model handlers
        def ml_model_serializer(model):
            return {
                "__custom_type__": "MLModel",
                "model_id": model.model_id,
                "model_type": model.model_type,
                "parameters": model.parameters,
                "weights": model.weights.tolist(),  # Convert numpy to list
                "training_history": model.training_history,
                "created_at": model.created_at.isoformat()
            }
        
        def ml_model_deserializer(data):
            return MockMLModel(
                model_id=data["model_id"],
                model_type=data["model_type"],
                parameters=data["parameters"],
                weights=np.array(data["weights"]),
                training_history=data["training_history"],
                created_at=datetime.fromisoformat(data["created_at"])
            )
        
        handler.register_type(MockMLModel, ml_model_serializer, ml_model_deserializer)
        
        # Test with ML model
        model = MockMLModel(
            model_id="model_123",
            model_type="neural_network",
            parameters={"learning_rate": 0.001, "batch_size": 32},
            weights=np.random.random((10, 5)),
            training_history=[{"epoch": 1, "loss": 0.5}],
            created_at=datetime.now()
        )
        
        # Serialize and deserialize
        serialized = handler.serialize(model)
        deserialized = handler.deserialize(serialized)
        
        assert isinstance(deserialized, MockMLModel)
        assert deserialized.model_id == model.model_id
        assert deserialized.model_type == model.model_type
        np.testing.assert_array_almost_equal(deserialized.weights, model.weights)
    
    def test_nested_custom_types(self, type_handler):
        """Test nested custom types"""
        handler = type_handler
        
        # Define nested custom classes
        class Playlist:
            def __init__(self, id, name, tracks):
                self.id = id
                self.name = name
                self.tracks = tracks
        
        class Track:
            def __init__(self, id, name, duration):
                self.id = id
                self.name = name
                self.duration = duration
        
        # Register handlers
        def track_serializer(track):
            return {
                "__custom_type__": "Track",
                "id": track.id,
                "name": track.name,
                "duration": track.duration
            }
        
        def track_deserializer(data):
            return Track(data["id"], data["name"], data["duration"])
        
        def playlist_serializer(playlist):
            return {
                "__custom_type__": "Playlist",
                "id": playlist.id,
                "name": playlist.name,
                "tracks": [handler.serialize(track) for track in playlist.tracks]
            }
        
        def playlist_deserializer(data):
            tracks = [handler.deserialize(track_data) for track_data in data["tracks"]]
            return Playlist(data["id"], data["name"], tracks)
        
        handler.register_type(Track, track_serializer, track_deserializer)
        handler.register_type(Playlist, playlist_serializer, playlist_deserializer)
        
        # Create nested structure
        tracks = [
            Track("track_1", "Song 1", 180),
            Track("track_2", "Song 2", 210)
        ]
        
        playlist = Playlist("playlist_1", "My Playlist", tracks)
        
        # Serialize and deserialize
        serialized = handler.serialize(playlist)
        deserialized = handler.deserialize(serialized)
        
        assert isinstance(deserialized, Playlist)
        assert deserialized.id == "playlist_1"
        assert len(deserialized.tracks) == 2
        assert all(isinstance(track, Track) for track in deserialized.tracks)


class TestSerializationMetrics:
    """Test serialization metrics"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Metrics collector fixture"""
        return SerializationMetrics(
            enable_timing=True,
            enable_size_tracking=True,
            enable_error_tracking=True
        )
    
    def test_timing_metrics(self, metrics_collector):
        """Test timing metrics collection"""
        metrics = metrics_collector
        
        # Simulate serialization with timing
        with metrics.time_operation("json_serialization"):
            time.sleep(0.01)  # Simulate work
            data = {"key": "value"}
            json.dumps(data)
        
        timing_stats = metrics.get_timing_stats()
        
        assert "json_serialization" in timing_stats
        assert timing_stats["json_serialization"]["count"] > 0
        assert timing_stats["json_serialization"]["total_time"] > 0
        assert timing_stats["json_serialization"]["average_time"] > 0
    
    def test_size_metrics(self, metrics_collector):
        """Test size metrics collection"""
        metrics = metrics_collector
        
        # Track different data sizes
        test_cases = [
            ("small", b"small data"),
            ("medium", b"x" * 1000),
            ("large", b"y" * 10000)
        ]
        
        for name, data in test_cases:
            compressed = gzip.compress(data)
            metrics.record_size_metrics(
                operation=f"{name}_compression",
                input_size=len(data),
                output_size=len(compressed)
            )
        
        size_stats = metrics.get_size_stats()
        
        assert "small_compression" in size_stats
        assert "medium_compression" in size_stats
        assert "large_compression" in size_stats
        
        # Check compression ratios
        for name in ["small_compression", "medium_compression", "large_compression"]:
            assert "compression_ratio" in size_stats[name]
            assert size_stats[name]["compression_ratio"] > 0
    
    def test_error_tracking(self, metrics_collector):
        """Test error tracking"""
        metrics = metrics_collector
        
        # Simulate various errors
        error_types = [
            ("json_decode_error", "Invalid JSON"),
            ("pickle_error", "Pickle protocol error"),
            ("compression_error", "Compression failed")
        ]
        
        for error_type, message in error_types:
            metrics.record_error(error_type, message)
            metrics.record_error(error_type, message)  # Record twice
        
        error_stats = metrics.get_error_stats()
        
        for error_type, _ in error_types:
            assert error_type in error_stats
            assert error_stats[error_type]["count"] == 2
            assert "last_occurrence" in error_stats[error_type]
    
    def test_performance_analysis(self, metrics_collector):
        """Test performance analysis"""
        metrics = metrics_collector
        
        # Generate varied performance data
        import random
        
        for i in range(100):
            operation_time = random.uniform(0.001, 0.1)
            input_size = random.randint(100, 10000)
            output_size = random.randint(50, input_size)
            
            with metrics.time_operation("varied_operation"):
                time.sleep(operation_time / 1000)  # Small delay
            
            metrics.record_size_metrics(
                "varied_operation",
                input_size,
                output_size
            )
        
        # Analyze performance
        analysis = metrics.analyze_performance()
        
        assert "throughput_ops_per_second" in analysis
        assert "average_compression_ratio" in analysis
        assert "performance_trends" in analysis
        assert analysis["throughput_ops_per_second"] > 0


class TestSerializationIntegration:
    """Integration tests for complete serialization system"""
    
    def test_complete_serialization_workflow(self):
        """Test complete serialization workflow"""
        # Initialize serialization manager
        manager = SerializationManager(
            default_format="json",
            enable_compression=True,
            compression_threshold=100,
            enable_metrics=True
        )
        
        # Register custom types
        def track_serializer(track):
            return asdict(track)
        
        def track_deserializer(data):
            # Convert datetime string back to datetime
            if isinstance(data.get("release_date"), str):
                data["release_date"] = datetime.fromisoformat(data["release_date"])
            return MockSpotifyTrack(**data)
        
        manager.register_custom_type(MockSpotifyTrack, track_serializer, track_deserializer)
        
        # Complex test data
        test_data = {
            "user_id": "user_123",
            "tracks": [
                MockSpotifyTrack(
                    id=f"track_{i}",
                    name=f"Song {i}",
                    artists=[f"Artist {i}"],
                    duration_ms=180000 + i * 1000,
                    popularity=50 + i,
                    audio_features={"energy": 0.5 + i * 0.1},
                    release_date=datetime.now() - timedelta(days=i)
                )
                for i in range(5)
            ],
            "preferences": {
                "genres": ["rock", "pop", "jazz"],
                "audio_features": {
                    "energy": {"min": 0.3, "max": 0.9},
                    "valence": {"min": 0.2, "max": 0.8}
                }
            },
            "timestamp": datetime.now()
        }
        
        # Serialize
        serialized = manager.serialize(test_data)
        assert isinstance(serialized, (str, bytes))
        
        # Deserialize
        deserialized = manager.deserialize(serialized)
        
        # Verify integrity
        assert deserialized["user_id"] == "user_123"
        assert len(deserialized["tracks"]) == 5
        assert all(isinstance(track, MockSpotifyTrack) for track in deserialized["tracks"])
        assert deserialized["preferences"]["genres"] == ["rock", "pop", "jazz"]
        
        # Check metrics
        metrics = manager.get_metrics()
        assert "serialization_time" in metrics
        assert "deserialization_time" in metrics
        assert "total_size_bytes" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
