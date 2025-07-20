"""
Sérialiseurs Avancés pour le Système de Cache
============================================

Suite complète de sérialiseurs optimisés pour différents types de données
avec compression, chiffrement et validation d'intégrité intégrés.

Fonctionnalités:
- Sérialisation adaptative selon le type de données
- Compression intelligente avec algorithmes multiples
- Chiffrement AES-256 pour données sensibles
- Validation d'intégrité avec checksums
- Support des types complexes (NumPy, Pandas, etc.)
- Optimisation automatique selon la taille et fréquence d'accès

Auteurs: Équipe Spotify AI Agent - Direction technique Fahed Mlaiel
"""

import json
import pickle
import gzip
import lz4.frame
import zstd
import hashlib
import base64
import struct
from typing import Any, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .exceptions import CacheSerializationError, CacheCompressionError, CacheSecurityError


class CompressionAlgorithm(Enum):
    """Algorithmes de compression supportés"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"


class SerializationFormat(Enum):
    """Formats de sérialisation supportés"""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    AVRO = "avro"
    PROTOBUF = "protobuf"


@dataclass
class SerializationMetadata:
    """Métadonnées de sérialisation"""
    format: SerializationFormat
    compression: CompressionAlgorithm
    compressed_size: int
    uncompressed_size: int
    checksum: str
    encrypted: bool = False
    compression_ratio: float = 0.0
    serialization_time: float = 0.0
    
    def __post_init__(self):
        if self.uncompressed_size > 0:
            self.compression_ratio = self.compressed_size / self.uncompressed_size


class BaseSerializer(ABC):
    """Interface de base pour tous les sérialiseurs"""
    
    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Sérialise un objet en bytes"""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Désérialise des bytes en objet"""
        pass
    
    @abstractmethod
    def get_format(self) -> SerializationFormat:
        """Retourne le format de sérialisation"""
        pass
    
    def calculate_checksum(self, data: bytes) -> str:
        """Calcule un checksum SHA-256 des données"""
        return hashlib.sha256(data).hexdigest()
    
    def validate_checksum(self, data: bytes, expected_checksum: str) -> bool:
        """Valide le checksum des données"""
        return self.calculate_checksum(data) == expected_checksum


class JSONSerializer(BaseSerializer):
    """Sérialiseur JSON optimisé avec support des types étendus"""
    
    def __init__(self, ensure_ascii: bool = False, sort_keys: bool = True):
        self.ensure_ascii = ensure_ascii
        self.sort_keys = sort_keys
    
    def serialize(self, obj: Any) -> bytes:
        """Sérialise en JSON avec support des types Python étendus"""
        try:
            json_str = json.dumps(
                obj, 
                default=self._json_serializer_hook,
                ensure_ascii=self.ensure_ascii,
                sort_keys=self.sort_keys,
                separators=(',', ':')  # Compact JSON
            )
            return json_str.encode('utf-8')
        except (TypeError, ValueError) as e:
            raise CacheSerializationError(
                "serialize", "json", type(obj).__name__
            ) from e
    
    def deserialize(self, data: bytes) -> Any:
        """Désérialise du JSON avec reconstruction des types"""
        try:
            json_str = data.decode('utf-8')
            return json.loads(json_str, object_hook=self._json_deserializer_hook)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise CacheSerializationError(
                "deserialize", "json", "bytes"
            ) from e
    
    def get_format(self) -> SerializationFormat:
        return SerializationFormat.JSON
    
    def _json_serializer_hook(self, obj: Any) -> Any:
        """Hook pour sérialiser des types non-JSON natifs"""
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape
            }
        elif isinstance(obj, pd.DataFrame):
            return {
                "__pandas_dataframe__": True,
                "data": obj.to_dict('records'),
                "columns": obj.columns.tolist(),
                "index": obj.index.tolist()
            }
        elif hasattr(obj, '__dict__'):
            # Support des objets personnalisés
            return {
                "__object__": True,
                "class": obj.__class__.__name__,
                "data": obj.__dict__
            }
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _json_deserializer_hook(self, dct: Dict[str, Any]) -> Any:
        """Hook pour désérialiser des types spéciaux"""
        if "__numpy_array__" in dct:
            return np.array(dct["data"], dtype=dct["dtype"]).reshape(dct["shape"])
        elif "__pandas_dataframe__" in dct:
            df = pd.DataFrame(dct["data"])
            df.columns = dct["columns"]
            df.index = dct["index"]
            return df
        
        return dct


class PickleSerializer(BaseSerializer):
    """Sérialiseur Pickle optimisé et sécurisé"""
    
    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol
    
    def serialize(self, obj: Any) -> bytes:
        """Sérialise avec Pickle"""
        try:
            return pickle.dumps(obj, protocol=self.protocol)
        except (pickle.PicklingError, TypeError) as e:
            raise CacheSerializationError(
                "serialize", "pickle", type(obj).__name__
            ) from e
    
    def deserialize(self, data: bytes) -> Any:
        """Désérialise avec Pickle"""
        try:
            return pickle.loads(data)
        except (pickle.UnpicklingError, EOFError) as e:
            raise CacheSerializationError(
                "deserialize", "pickle", "bytes"
            ) from e
    
    def get_format(self) -> SerializationFormat:
        return SerializationFormat.PICKLE


class MsgPackSerializer(BaseSerializer):
    """Sérialiseur MessagePack pour performance optimale"""
    
    def __init__(self):
        try:
            import msgpack
            self.msgpack = msgpack
        except ImportError:
            raise ImportError("msgpack-python required for MsgPackSerializer")
    
    def serialize(self, obj: Any) -> bytes:
        """Sérialise avec MessagePack"""
        try:
            return self.msgpack.packb(obj, use_bin_type=True)
        except (TypeError, ValueError) as e:
            raise CacheSerializationError(
                "serialize", "msgpack", type(obj).__name__
            ) from e
    
    def deserialize(self, data: bytes) -> Any:
        """Désérialise avec MessagePack"""
        try:
            return self.msgpack.unpackb(data, raw=False, strict_map_key=False)
        except (self.msgpack.exceptions.ExtraData, ValueError) as e:
            raise CacheSerializationError(
                "deserialize", "msgpack", "bytes"
            ) from e
    
    def get_format(self) -> SerializationFormat:
        return SerializationFormat.MSGPACK


class CompressionUtils:
    """Utilitaires de compression avancés"""
    
    @staticmethod
    def compress(data: bytes, algorithm: CompressionAlgorithm, 
                level: int = 6) -> bytes:
        """Compresse les données avec l'algorithme spécifié"""
        try:
            if algorithm == CompressionAlgorithm.NONE:
                return data
            elif algorithm == CompressionAlgorithm.GZIP:
                return gzip.compress(data, compresslevel=level)
            elif algorithm == CompressionAlgorithm.LZ4:
                return lz4.frame.compress(data, compression_level=level)
            elif algorithm == CompressionAlgorithm.ZSTD:
                return zstd.compress(data, level=level)
            else:
                raise ValueError(f"Unsupported compression algorithm: {algorithm}")
        except Exception as e:
            raise CacheCompressionError(
                "compress", algorithm.value, len(data)
            ) from e
    
    @staticmethod
    def decompress(data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Décompresse les données"""
        try:
            if algorithm == CompressionAlgorithm.NONE:
                return data
            elif algorithm == CompressionAlgorithm.GZIP:
                return gzip.decompress(data)
            elif algorithm == CompressionAlgorithm.LZ4:
                return lz4.frame.decompress(data)
            elif algorithm == CompressionAlgorithm.ZSTD:
                return zstd.decompress(data)
            else:
                raise ValueError(f"Unsupported compression algorithm: {algorithm}")
        except Exception as e:
            raise CacheCompressionError(
                "decompress", algorithm.value
            ) from e
    
    @staticmethod
    def choose_best_compression(data: bytes, 
                              algorithms: list = None) -> Tuple[CompressionAlgorithm, bytes]:
        """Choisit le meilleur algorithme de compression pour les données"""
        if algorithms is None:
            algorithms = [
                CompressionAlgorithm.LZ4,
                CompressionAlgorithm.ZSTD,
                CompressionAlgorithm.GZIP
            ]
        
        best_algo = CompressionAlgorithm.NONE
        best_compressed = data
        best_ratio = 1.0
        
        for algo in algorithms:
            try:
                compressed = CompressionUtils.compress(data, algo)
                ratio = len(compressed) / len(data)
                
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_algo = algo
                    best_compressed = compressed
            except Exception:
                continue
        
        return best_algo, best_compressed


class EncryptionUtils:
    """Utilitaires de chiffrement pour données sensibles"""
    
    def __init__(self, password: str):
        self.password = password.encode('utf-8')
        self._setup_encryption()
    
    def _setup_encryption(self):
        """Configure le chiffrement avec dérivation de clé sécurisée"""
        salt = b'spotify_ai_cache_salt'  # En production, utiliser un salt aléatoire
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: bytes) -> bytes:
        """Chiffre les données"""
        try:
            return self.cipher.encrypt(data)
        except Exception as e:
            raise CacheSecurityError(
                f"Encryption failed: {e}", operation="encrypt"
            )
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Déchiffre les données"""
        try:
            return self.cipher.decrypt(encrypted_data)
        except Exception as e:
            raise CacheSecurityError(
                f"Decryption failed: {e}", operation="decrypt"
            )


class CompressionSerializer(BaseSerializer):
    """Sérialiseur avec compression automatique intelligente"""
    
    def __init__(self, base_serializer: BaseSerializer,
                 compression_threshold: int = 1024,
                 auto_compression: bool = True):
        self.base_serializer = base_serializer
        self.compression_threshold = compression_threshold
        self.auto_compression = auto_compression
        self.compression_stats = {}
    
    def serialize(self, obj: Any) -> bytes:
        """Sérialise avec compression intelligente"""
        # Sérialisation de base
        base_data = self.base_serializer.serialize(obj)
        
        # Décision de compression
        if not self.auto_compression or len(base_data) < self.compression_threshold:
            return self._pack_data(base_data, CompressionAlgorithm.NONE, base_data)
        
        # Sélection du meilleur algorithme
        best_algo, compressed_data = CompressionUtils.choose_best_compression(base_data)
        
        # Statistiques de compression
        self._update_compression_stats(best_algo, len(base_data), len(compressed_data))
        
        return self._pack_data(base_data, best_algo, compressed_data)
    
    def deserialize(self, data: bytes) -> Any:
        """Désérialise avec décompression automatique"""
        # Extraction des métadonnées
        original_data, algorithm = self._unpack_data(data)
        
        # Décompression si nécessaire
        if algorithm != CompressionAlgorithm.NONE:
            original_data = CompressionUtils.decompress(original_data, algorithm)
        
        # Désérialisation de base
        return self.base_serializer.deserialize(original_data)
    
    def get_format(self) -> SerializationFormat:
        return self.base_serializer.get_format()
    
    def _pack_data(self, original: bytes, algorithm: CompressionAlgorithm, 
                   processed: bytes) -> bytes:
        """Emballe les données avec métadonnées de compression"""
        # Format: [header_size:4][header][data]
        header = {
            "algorithm": algorithm.value,
            "original_size": len(original),
            "compressed_size": len(processed),
            "checksum": self.calculate_checksum(original)
        }
        
        header_bytes = json.dumps(header).encode('utf-8')
        header_size = struct.pack('<I', len(header_bytes))
        
        return header_size + header_bytes + processed
    
    def _unpack_data(self, data: bytes) -> Tuple[bytes, CompressionAlgorithm]:
        """Dépaquetage des données avec métadonnées"""
        # Lecture de la taille du header
        header_size = struct.unpack('<I', data[:4])[0]
        
        # Lecture du header
        header_bytes = data[4:4+header_size]
        header = json.loads(header_bytes.decode('utf-8'))
        
        # Extraction des données
        processed_data = data[4+header_size:]
        algorithm = CompressionAlgorithm(header["algorithm"])
        
        return processed_data, algorithm
    
    def _update_compression_stats(self, algorithm: CompressionAlgorithm, 
                                 original_size: int, compressed_size: int):
        """Met à jour les statistiques de compression"""
        if algorithm.value not in self.compression_stats:
            self.compression_stats[algorithm.value] = {
                "usage_count": 0,
                "total_original_size": 0,
                "total_compressed_size": 0,
                "avg_compression_ratio": 0.0
            }
        
        stats = self.compression_stats[algorithm.value]
        stats["usage_count"] += 1
        stats["total_original_size"] += original_size
        stats["total_compressed_size"] += compressed_size
        stats["avg_compression_ratio"] = (
            stats["total_compressed_size"] / stats["total_original_size"]
        )


class EncryptedSerializer(BaseSerializer):
    """Sérialiseur avec chiffrement pour données sensibles"""
    
    def __init__(self, base_serializer: BaseSerializer, encryption_key: str):
        self.base_serializer = base_serializer
        self.encryption_utils = EncryptionUtils(encryption_key)
    
    def serialize(self, obj: Any) -> bytes:
        """Sérialise et chiffre les données"""
        # Sérialisation de base
        base_data = self.base_serializer.serialize(obj)
        
        # Chiffrement
        encrypted_data = self.encryption_utils.encrypt(base_data)
        
        # Métadonnées de chiffrement
        return self._pack_encrypted_data(encrypted_data)
    
    def deserialize(self, data: bytes) -> Any:
        """Déchiffre et désérialise les données"""
        # Dépackage
        encrypted_data = self._unpack_encrypted_data(data)
        
        # Déchiffrement
        decrypted_data = self.encryption_utils.decrypt(encrypted_data)
        
        # Désérialisation de base
        return self.base_serializer.deserialize(decrypted_data)
    
    def get_format(self) -> SerializationFormat:
        return self.base_serializer.get_format()
    
    def _pack_encrypted_data(self, encrypted_data: bytes) -> bytes:
        """Emballe les données chiffrées avec métadonnées"""
        header = {
            "encrypted": True,
            "encryption_version": "1.0",
            "data_size": len(encrypted_data)
        }
        
        header_bytes = json.dumps(header).encode('utf-8')
        header_size = struct.pack('<I', len(header_bytes))
        
        return header_size + header_bytes + encrypted_data
    
    def _unpack_encrypted_data(self, data: bytes) -> bytes:
        """Dépaquetage des données chiffrées"""
        # Lecture de la taille du header
        header_size = struct.unpack('<I', data[:4])[0]
        
        # Lecture du header (pour validation)
        header_bytes = data[4:4+header_size]
        header = json.loads(header_bytes.decode('utf-8'))
        
        if not header.get("encrypted", False):
            raise CacheSecurityError("Data is not encrypted", operation="decrypt")
        
        # Extraction des données chiffrées
        return data[4+header_size:]


class AdaptiveSerializer:
    """Sérialiseur adaptatif qui choisit automatiquement le meilleur format"""
    
    def __init__(self, enable_compression: bool = True, 
                 enable_encryption: bool = False, encryption_key: str = None):
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        
        # Sérialiseurs de base
        self.serializers = {
            SerializationFormat.JSON: JSONSerializer(),
            SerializationFormat.PICKLE: PickleSerializer(),
            SerializationFormat.MSGPACK: MsgPackSerializer()
        }
        
        # Configuration de la compression et du chiffrement
        if enable_compression:
            for format_type, serializer in self.serializers.items():
                self.serializers[format_type] = CompressionSerializer(serializer)
        
        if enable_encryption and encryption_key:
            for format_type, serializer in self.serializers.items():
                self.serializers[format_type] = EncryptedSerializer(serializer, encryption_key)
        
        # Statistiques d'utilisation pour l'apprentissage
        self.usage_stats = {}
    
    def serialize(self, obj: Any) -> Tuple[bytes, SerializationMetadata]:
        """Sérialise avec le format optimal"""
        best_format = self._choose_best_format(obj)
        serializer = self.serializers[best_format]
        
        import time
        start_time = time.time()
        
        serialized_data = serializer.serialize(obj)
        
        serialization_time = time.time() - start_time
        
        # Création des métadonnées
        metadata = SerializationMetadata(
            format=best_format,
            compression=CompressionAlgorithm.ZSTD if self.enable_compression else CompressionAlgorithm.NONE,
            compressed_size=len(serialized_data),
            uncompressed_size=len(str(obj)),  # Approximation
            checksum=serializer.calculate_checksum(serialized_data),
            encrypted=self.enable_encryption,
            serialization_time=serialization_time
        )
        
        return serialized_data, metadata
    
    def deserialize(self, data: bytes, metadata: SerializationMetadata) -> Any:
        """Désérialise selon les métadonnées"""
        serializer = self.serializers[metadata.format]
        
        # Validation de l'intégrité
        if not serializer.validate_checksum(data, metadata.checksum):
            raise CacheSerializationError(
                "deserialize", metadata.format.value, "checksum_validation_failed"
            )
        
        return serializer.deserialize(data)
    
    def _choose_best_format(self, obj: Any) -> SerializationFormat:
        """Choisit le meilleur format selon le type d'objet"""
        obj_type = type(obj).__name__
        
        # Règles heuristiques
        if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
            return SerializationFormat.JSON
        elif isinstance(obj, (np.ndarray, pd.DataFrame)):
            return SerializationFormat.PICKLE
        else:
            # Pour les objets complexes, utiliser le format le plus performant observé
            if obj_type in self.usage_stats:
                best_format = min(
                    self.usage_stats[obj_type].items(),
                    key=lambda x: x[1]['avg_serialization_time']
                )[0]
                return SerializationFormat(best_format)
            else:
                return SerializationFormat.PICKLE  # Par défaut
    
    def update_usage_stats(self, obj_type: str, format_used: SerializationFormat,
                          serialization_time: float, data_size: int):
        """Met à jour les statistiques d'utilisation"""
        if obj_type not in self.usage_stats:
            self.usage_stats[obj_type] = {}
        
        if format_used.value not in self.usage_stats[obj_type]:
            self.usage_stats[obj_type][format_used.value] = {
                "usage_count": 0,
                "total_time": 0.0,
                "total_size": 0,
                "avg_serialization_time": 0.0,
                "avg_size": 0
            }
        
        stats = self.usage_stats[obj_type][format_used.value]
        stats["usage_count"] += 1
        stats["total_time"] += serialization_time
        stats["total_size"] += data_size
        stats["avg_serialization_time"] = stats["total_time"] / stats["usage_count"]
        stats["avg_size"] = stats["total_size"] / stats["usage_count"]
