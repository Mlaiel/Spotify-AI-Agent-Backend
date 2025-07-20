"""
Sérialiseurs avancés - Spotify AI Agent
Système de sérialisation multi-format avec support d'optimisation
"""

import json
import yaml
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union, Type, Protocol, BinaryIO, TextIO
from datetime import datetime, date, time, timezone
from decimal import Decimal
from enum import Enum
from uuid import UUID
from pathlib import Path
import pickle
import msgpack
from dataclasses import dataclass, asdict
import base64
import gzip
import zlib
from io import StringIO, BytesIO

from pydantic import BaseModel
from pydantic.json import pydantic_encoder

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

try:
    import ujson
    HAS_UJSON = True
except ImportError:
    HAS_UJSON = False

try:
    import protobuf
    HAS_PROTOBUF = True
except ImportError:
    HAS_PROTOBUF = False


class SerializationFormat(str, Enum):
    """Formats de sérialisation supportés"""
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"
    CSV = "csv"
    TOML = "toml"
    INI = "ini"
    BINARY = "binary"


class CompressionType(str, Enum):
    """Types de compression supportés"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    BROTLI = "brotli"
    LZ4 = "lz4"


class SerializationOptions(BaseModel):
    """Options de sérialisation configurables"""
    format: SerializationFormat = SerializationFormat.JSON
    compression: CompressionType = CompressionType.NONE
    indent: Optional[int] = 2
    ensure_ascii: bool = False
    sort_keys: bool = False
    include_metadata: bool = True
    include_schema: bool = False
    encoding: str = "utf-8"
    binary_encoding: str = "base64"
    datetime_format: str = "iso"
    decimal_places: Optional[int] = None
    exclude_none: bool = False
    exclude_defaults: bool = False
    use_enum_values: bool = True
    optimize_for_size: bool = False
    optimize_for_speed: bool = False
    validate_on_deserialize: bool = True
    preserve_order: bool = True
    max_depth: int = 100
    
    class Config:
        use_enum_values = True


@dataclass
class SerializationResult:
    """Résultat de sérialisation"""
    data: Union[str, bytes]
    format: SerializationFormat
    size_bytes: int
    compression_ratio: Optional[float] = None
    serialization_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SerializationError(Exception):
    """Erreur de sérialisation"""
    pass


class DeserializationError(Exception):
    """Erreur de désérialisation"""
    pass


class CustomEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour types avancés"""
    
    def default(self, obj):
        """Encode les types personnalisés"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, time):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        elif hasattr(obj, 'dict'):  # Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        
        return super().default(obj)


class BaseSerializer:
    """Sérialiseur de base avec fonctionnalités communes"""
    
    def __init__(self, options: Optional[SerializationOptions] = None):
        self.options = options or SerializationOptions()
        self._compressors = {
            CompressionType.GZIP: self._compress_gzip,
            CompressionType.ZLIB: self._compress_zlib,
        }
        self._decompressors = {
            CompressionType.GZIP: self._decompress_gzip,
            CompressionType.ZLIB: self._decompress_zlib,
        }
    
    def _prepare_data(self, data: Any) -> Any:
        """Prépare les données pour la sérialisation"""
        if isinstance(data, BaseModel):
            return data.dict(
                exclude_none=self.options.exclude_none,
                exclude_defaults=self.options.exclude_defaults,
                by_alias=True
            )
        
        return data
    
    def _compress_data(self, data: Union[str, bytes]) -> bytes:
        """Compresse les données selon l'option configurée"""
        if self.options.compression == CompressionType.NONE:
            return data.encode(self.options.encoding) if isinstance(data, str) else data
        
        if isinstance(data, str):
            data = data.encode(self.options.encoding)
        
        compressor = self._compressors.get(self.options.compression)
        if compressor:
            return compressor(data)
        
        raise SerializationError(f"Compression non supportée: {self.options.compression}")
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Décompresse les données"""
        if self.options.compression == CompressionType.NONE:
            return data
        
        decompressor = self._decompressors.get(self.options.compression)
        if decompressor:
            return decompressor(data)
        
        raise DeserializationError(f"Décompression non supportée: {self.options.compression}")
    
    def _compress_gzip(self, data: bytes) -> bytes:
        """Compression GZIP"""
        return gzip.compress(data)
    
    def _decompress_gzip(self, data: bytes) -> bytes:
        """Décompression GZIP"""
        return gzip.decompress(data)
    
    def _compress_zlib(self, data: bytes) -> bytes:
        """Compression ZLIB"""
        return zlib.compress(data)
    
    def _decompress_zlib(self, data: bytes) -> bytes:
        """Décompression ZLIB"""
        return zlib.decompress(data)


class JsonSerializer(BaseSerializer):
    """Sérialiseur JSON optimisé"""
    
    def serialize(self, data: Any) -> SerializationResult:
        """Sérialise en JSON"""
        import time
        start_time = time.time()
        
        prepared_data = self._prepare_data(data)
        
        # Choisir le meilleur encodeur JSON disponible
        if self.options.optimize_for_speed and HAS_ORJSON:
            json_str = orjson.dumps(
                prepared_data,
                option=orjson.OPT_INDENT_2 if self.options.indent else 0
            ).decode('utf-8')
        elif HAS_UJSON and self.options.optimize_for_speed:
            json_str = ujson.dumps(
                prepared_data,
                indent=self.options.indent,
                ensure_ascii=self.options.ensure_ascii
            )
        else:
            json_str = json.dumps(
                prepared_data,
                cls=CustomEncoder,
                indent=self.options.indent,
                ensure_ascii=self.options.ensure_ascii,
                sort_keys=self.options.sort_keys
            )
        
        # Compression si nécessaire
        final_data = self._compress_data(json_str)
        
        serialization_time = time.time() - start_time
        original_size = len(json_str.encode(self.options.encoding))
        
        return SerializationResult(
            data=final_data,
            format=SerializationFormat.JSON,
            size_bytes=len(final_data),
            compression_ratio=len(final_data) / original_size if self.options.compression != CompressionType.NONE else None,
            serialization_time=serialization_time,
            metadata={
                'encoder': 'orjson' if HAS_ORJSON and self.options.optimize_for_speed else 'json',
                'original_size': original_size
            }
        )
    
    def deserialize(self, data: Union[str, bytes], target_type: Optional[Type] = None) -> Any:
        """Désérialise depuis JSON"""
        if isinstance(data, bytes):
            # Décompression si nécessaire
            decompressed = self._decompress_data(data)
            json_str = decompressed.decode(self.options.encoding)
        else:
            json_str = data
        
        # Choisir le meilleur décodeur JSON disponible
        if HAS_ORJSON and self.options.optimize_for_speed:
            result = orjson.loads(json_str)
        elif HAS_UJSON and self.options.optimize_for_speed:
            result = ujson.loads(json_str)
        else:
            result = json.loads(json_str)
        
        # Conversion vers le type cible si spécifié
        if target_type and issubclass(target_type, BaseModel):
            if self.options.validate_on_deserialize:
                return target_type.parse_obj(result)
            else:
                return target_type.construct(result)
        
        return result


class YamlSerializer(BaseSerializer):
    """Sérialiseur YAML"""
    
    def serialize(self, data: Any) -> SerializationResult:
        """Sérialise en YAML"""
        import time
        start_time = time.time()
        
        prepared_data = self._prepare_data(data)
        
        yaml_str = yaml.dump(
            prepared_data,
            default_flow_style=False,
            sort_keys=self.options.sort_keys,
            indent=self.options.indent,
            encoding=None,  # Return string, not bytes
            allow_unicode=not self.options.ensure_ascii
        )
        
        final_data = self._compress_data(yaml_str)
        serialization_time = time.time() - start_time
        
        return SerializationResult(
            data=final_data,
            format=SerializationFormat.YAML,
            size_bytes=len(final_data),
            serialization_time=serialization_time
        )
    
    def deserialize(self, data: Union[str, bytes], target_type: Optional[Type] = None) -> Any:
        """Désérialise depuis YAML"""
        if isinstance(data, bytes):
            decompressed = self._decompress_data(data)
            yaml_str = decompressed.decode(self.options.encoding)
        else:
            yaml_str = data
        
        result = yaml.safe_load(yaml_str)
        
        if target_type and issubclass(target_type, BaseModel):
            return target_type.parse_obj(result)
        
        return result


class XmlSerializer(BaseSerializer):
    """Sérialiseur XML"""
    
    def serialize(self, data: Any) -> SerializationResult:
        """Sérialise en XML"""
        import time
        start_time = time.time()
        
        prepared_data = self._prepare_data(data)
        root = self._dict_to_xml(prepared_data, "root")
        
        xml_str = ET.tostring(root, encoding=self.options.encoding, xml_declaration=True)
        final_data = self._compress_data(xml_str)
        
        serialization_time = time.time() - start_time
        
        return SerializationResult(
            data=final_data,
            format=SerializationFormat.XML,
            size_bytes=len(final_data),
            serialization_time=serialization_time
        )
    
    def _dict_to_xml(self, data: Any, tag_name: str = "item") -> ET.Element:
        """Convertit un dictionnaire en élément XML"""
        element = ET.Element(tag_name)
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    child = self._dict_to_xml(value, str(key))
                    element.append(child)
                else:
                    child = ET.SubElement(element, str(key))
                    child.text = str(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                child = self._dict_to_xml(item, f"item_{i}")
                element.append(child)
        else:
            element.text = str(data)
        
        return element
    
    def deserialize(self, data: Union[str, bytes], target_type: Optional[Type] = None) -> Any:
        """Désérialise depuis XML"""
        if isinstance(data, bytes):
            decompressed = self._decompress_data(data)
            xml_str = decompressed.decode(self.options.encoding)
        else:
            xml_str = data
        
        root = ET.fromstring(xml_str)
        result = self._xml_to_dict(root)
        
        if target_type and issubclass(target_type, BaseModel):
            return target_type.parse_obj(result)
        
        return result
    
    def _xml_to_dict(self, element: ET.Element) -> Any:
        """Convertit un élément XML en dictionnaire"""
        result = {}
        
        # Traiter les attributs
        if element.attrib:
            result.update(element.attrib)
        
        # Traiter le texte
        if element.text and element.text.strip():
            if len(result) == 0:
                return element.text.strip()
            result['_text'] = element.text.strip()
        
        # Traiter les enfants
        children = {}
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in children:
                if not isinstance(children[child.tag], list):
                    children[child.tag] = [children[child.tag]]
                children[child.tag].append(child_data)
            else:
                children[child.tag] = child_data
        
        result.update(children)
        return result


class MsgPackSerializer(BaseSerializer):
    """Sérialiseur MessagePack pour performance binaire"""
    
    def serialize(self, data: Any) -> SerializationResult:
        """Sérialise en MessagePack"""
        import time
        start_time = time.time()
        
        prepared_data = self._prepare_data(data)
        
        # Encodage personnalisé pour types non supportés
        def default_encoder(obj):
            if isinstance(obj, datetime):
                return {'__datetime__': obj.isoformat()}
            elif isinstance(obj, UUID):
                return {'__uuid__': str(obj)}
            elif isinstance(obj, Decimal):
                return {'__decimal__': str(obj)}
            return obj
        
        msgpack_bytes = msgpack.packb(
            prepared_data,
            default=default_encoder,
            use_bin_type=True,
            strict_types=True
        )
        
        final_data = self._compress_data(msgpack_bytes)
        serialization_time = time.time() - start_time
        
        return SerializationResult(
            data=final_data,
            format=SerializationFormat.MSGPACK,
            size_bytes=len(final_data),
            serialization_time=serialization_time
        )
    
    def deserialize(self, data: bytes, target_type: Optional[Type] = None) -> Any:
        """Désérialise depuis MessagePack"""
        decompressed = self._decompress_data(data)
        
        def object_hook(obj):
            if '__datetime__' in obj:
                return datetime.fromisoformat(obj['__datetime__'])
            elif '__uuid__' in obj:
                return UUID(obj['__uuid__'])
            elif '__decimal__' in obj:
                return Decimal(obj['__decimal__'])
            return obj
        
        result = msgpack.unpackb(
            decompressed,
            object_hook=object_hook,
            raw=False,
            strict_map_key=False
        )
        
        if target_type and issubclass(target_type, BaseModel):
            return target_type.parse_obj(result)
        
        return result


class PickleSerializer(BaseSerializer):
    """Sérialiseur Pickle pour objets Python natifs"""
    
    def serialize(self, data: Any) -> SerializationResult:
        """Sérialise en Pickle"""
        import time
        start_time = time.time()
        
        # Pickle conserve les objets Python natifs
        pickle_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        final_data = self._compress_data(pickle_bytes)
        
        serialization_time = time.time() - start_time
        
        return SerializationResult(
            data=final_data,
            format=SerializationFormat.PICKLE,
            size_bytes=len(final_data),
            serialization_time=serialization_time,
            metadata={'pickle_protocol': pickle.HIGHEST_PROTOCOL}
        )
    
    def deserialize(self, data: bytes, target_type: Optional[Type] = None) -> Any:
        """Désérialise depuis Pickle"""
        decompressed = self._decompress_data(data)
        return pickle.loads(decompressed)


class SerializerFactory:
    """Factory pour créer des sérialiseurs"""
    
    _serializers = {
        SerializationFormat.JSON: JsonSerializer,
        SerializationFormat.YAML: YamlSerializer,
        SerializationFormat.XML: XmlSerializer,
        SerializationFormat.MSGPACK: MsgPackSerializer,
        SerializationFormat.PICKLE: PickleSerializer,
    }
    
    @classmethod
    def create_serializer(cls, 
                         format_type: SerializationFormat,
                         options: Optional[SerializationOptions] = None) -> BaseSerializer:
        """Crée un sérialiseur pour le format spécifié"""
        serializer_class = cls._serializers.get(format_type)
        if not serializer_class:
            raise SerializationError(f"Format non supporté: {format_type}")
        
        return serializer_class(options)
    
    @classmethod
    def register_serializer(cls, format_type: SerializationFormat, serializer_class: Type[BaseSerializer]):
        """Enregistre un nouveau sérialiseur"""
        cls._serializers[format_type] = serializer_class
    
    @classmethod
    def get_optimal_format(cls, data_size: int, performance_priority: str = "balanced") -> SerializationFormat:
        """Suggère le format optimal selon les critères"""
        if performance_priority == "size":
            if data_size > 1024 * 1024:  # > 1MB
                return SerializationFormat.MSGPACK
            return SerializationFormat.JSON
        elif performance_priority == "speed":
            return SerializationFormat.MSGPACK if data_size > 1024 else SerializationFormat.JSON
        else:  # balanced
            return SerializationFormat.JSON


class MultiFormatSerializer:
    """Sérialiseur multi-format avec détection automatique"""
    
    def __init__(self, default_options: Optional[SerializationOptions] = None):
        self.default_options = default_options or SerializationOptions()
        self.serializers = {}
    
    def serialize(self, data: Any, 
                 format_type: Optional[SerializationFormat] = None,
                 options: Optional[SerializationOptions] = None) -> SerializationResult:
        """Sérialise avec détection automatique du format optimal"""
        if format_type is None:
            # Détection automatique basée sur le type de données
            format_type = self._detect_optimal_format(data)
        
        effective_options = options or self.default_options
        serializer = self._get_serializer(format_type, effective_options)
        
        return serializer.serialize(data)
    
    def deserialize(self, data: Union[str, bytes], 
                   format_type: Optional[SerializationFormat] = None,
                   target_type: Optional[Type] = None,
                   options: Optional[SerializationOptions] = None) -> Any:
        """Désérialise avec détection automatique du format"""
        if format_type is None:
            format_type = self._detect_format(data)
        
        effective_options = options or self.default_options
        serializer = self._get_serializer(format_type, effective_options)
        
        return serializer.deserialize(data, target_type)
    
    def _get_serializer(self, format_type: SerializationFormat, 
                       options: SerializationOptions) -> BaseSerializer:
        """Obtient ou crée un sérialiseur pour le format"""
        cache_key = (format_type, hash(options.json()))
        
        if cache_key not in self.serializers:
            self.serializers[cache_key] = SerializerFactory.create_serializer(format_type, options)
        
        return self.serializers[cache_key]
    
    def _detect_optimal_format(self, data: Any) -> SerializationFormat:
        """Détecte le format optimal pour les données"""
        if isinstance(data, (BaseModel, dict)):
            return SerializationFormat.JSON
        elif isinstance(data, list):
            return SerializationFormat.JSON
        else:
            return SerializationFormat.PICKLE
    
    def _detect_format(self, data: Union[str, bytes]) -> SerializationFormat:
        """Détecte le format des données sérialisées"""
        if isinstance(data, bytes):
            # Essayer de détecter les signatures binaires
            if data.startswith(b'\x80'):  # MessagePack
                return SerializationFormat.MSGPACK
            elif data.startswith(b'\x80\x02'):  # Pickle
                return SerializationFormat.PICKLE
            elif data.startswith(b'<?xml'):
                return SerializationFormat.XML
            else:
                # Peut être du JSON/YAML compressé
                try:
                    text = data.decode('utf-8')
                    return self._detect_text_format(text)
                except UnicodeDecodeError:
                    return SerializationFormat.PICKLE
        else:
            return self._detect_text_format(data)
    
    def _detect_text_format(self, text: str) -> SerializationFormat:
        """Détecte le format pour du texte"""
        text = text.strip()
        if text.startswith('{') or text.startswith('['):
            return SerializationFormat.JSON
        elif text.startswith('<?xml') or text.startswith('<'):
            return SerializationFormat.XML
        else:
            return SerializationFormat.YAML


# Instance globale par défaut
default_serializer = MultiFormatSerializer()

# Fonctions utilitaires
def serialize(data: Any, format_type: SerializationFormat = SerializationFormat.JSON, **kwargs) -> SerializationResult:
    """Fonction utilitaire pour sérialisation rapide"""
    options = SerializationOptions(**kwargs) if kwargs else None
    return default_serializer.serialize(data, format_type, options)

def deserialize(data: Union[str, bytes], target_type: Optional[Type] = None, 
               format_type: Optional[SerializationFormat] = None, **kwargs) -> Any:
    """Fonction utilitaire pour désérialisation rapide"""
    options = SerializationOptions(**kwargs) if kwargs else None
    return default_serializer.deserialize(data, format_type, target_type, options)


__all__ = [
    'SerializationFormat', 'CompressionType', 'SerializationOptions', 'SerializationResult',
    'BaseSerializer', 'JsonSerializer', 'YamlSerializer', 'XmlSerializer', 
    'MsgPackSerializer', 'PickleSerializer', 'SerializerFactory', 'MultiFormatSerializer',
    'serialize', 'deserialize', 'default_serializer'
]
