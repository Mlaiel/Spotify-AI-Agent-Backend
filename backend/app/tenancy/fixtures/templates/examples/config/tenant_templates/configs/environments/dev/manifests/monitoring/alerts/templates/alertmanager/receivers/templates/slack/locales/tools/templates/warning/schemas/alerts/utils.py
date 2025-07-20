"""
Utilitaires avancés - Spotify AI Agent  
Fonctions d'aide et outils pour la gestion des alertes
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable, TypeVar, Generic
from uuid import UUID, uuid4
from enum import Enum
import re
import json
import hashlib
import base64
import zlib
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel

from . import (
    BaseSchema, AlertLevel, AlertStatus, WarningCategory, Priority, Environment,
    NotificationChannel, EscalationLevel, CorrelationMethod, WorkflowStatus
)

T = TypeVar('T')


class CacheStrategy(str, Enum):
    """Stratégies de cache"""
    NONE = "none"
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


class SerializationFormat(str, Enum):
    """Formats de sérialisation"""
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    PROTOBUF = "protobuf"
    PICKLE = "pickle"


class CompressionType(str, Enum):
    """Types de compression"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    BROTLI = "brotli"


@dataclass
class ProcessingMetrics:
    """Métriques de traitement"""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    items_processed: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    memory_used_mb: Optional[float] = None
    
    def finish(self):
        """Finalise les métriques"""
        self.end_time = datetime.now(timezone.utc)
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000


class DataTransformer:
    """Transformateur de données"""
    
    @staticmethod
    def normalize_alert_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise les données d'alerte"""
        normalized = data.copy()
        
        # Normalisation des noms de champs
        field_mappings = {
            'alertname': 'name',
            'alertLevel': 'level',
            'alertStatus': 'status',
            'created': 'created_at',
            'updated': 'updated_at'
        }
        
        for old_key, new_key in field_mappings.items():
            if old_key in normalized:
                normalized[new_key] = normalized.pop(old_key)
        
        # Normalisation des valeurs
        if 'level' in normalized:
            normalized['level'] = DataTransformer._normalize_alert_level(normalized['level'])
        
        if 'status' in normalized:
            normalized['status'] = DataTransformer._normalize_alert_status(normalized['status'])
        
        # Normalisation des timestamps
        for time_field in ['created_at', 'updated_at', 'resolved_at']:
            if time_field in normalized:
                normalized[time_field] = DataTransformer._normalize_timestamp(
                    normalized[time_field]
                )
        
        return normalized
    
    @staticmethod
    def _normalize_alert_level(level: Union[str, int]) -> str:
        """Normalise le niveau d'alerte"""
        if isinstance(level, int):
            level_map = {0: 'info', 1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}
            return level_map.get(level, 'medium')
        
        level_str = str(level).lower()
        level_aliases = {
            'warn': 'warning',
            'error': 'high',
            'fatal': 'critical',
            'debug': 'info'
        }
        
        return level_aliases.get(level_str, level_str)
    
    @staticmethod
    def _normalize_alert_status(status: Union[str, int]) -> str:
        """Normalise le statut d'alerte"""
        if isinstance(status, int):
            status_map = {0: 'pending', 1: 'active', 2: 'resolved', 3: 'suppressed'}
            return status_map.get(status, 'pending')
        
        return str(status).lower()
    
    @staticmethod
    def _normalize_timestamp(timestamp: Union[str, int, datetime]) -> datetime:
        """Normalise un timestamp"""
        if isinstance(timestamp, datetime):
            return timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
        
        if isinstance(timestamp, (int, float)):
            # Unix timestamp
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        if isinstance(timestamp, str):
            # Parse ISO format
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            except ValueError:
                return datetime.now(timezone.utc)
        
        return datetime.now(timezone.utc)


class FilterEngine:
    """Moteur de filtrage avancé"""
    
    def __init__(self):
        self.filters: List[Callable] = []
    
    def add_filter(self, filter_func: Callable[[Any], bool]):
        """Ajoute un filtre"""
        self.filters.append(filter_func)
    
    def add_field_filter(self, field: str, value: Any, operator: str = 'eq'):
        """Ajoute un filtre sur un champ"""
        def field_filter(item):
            item_value = self._get_nested_value(item, field)
            return self._apply_operator(item_value, value, operator)
        
        self.add_filter(field_filter)
    
    def add_regex_filter(self, field: str, pattern: str):
        """Ajoute un filtre regex"""
        regex = re.compile(pattern)
        
        def regex_filter(item):
            item_value = str(self._get_nested_value(item, field, ''))
            return bool(regex.search(item_value))
        
        self.add_filter(regex_filter)
    
    def add_range_filter(self, field: str, min_value: Any = None, max_value: Any = None):
        """Ajoute un filtre de plage"""
        def range_filter(item):
            item_value = self._get_nested_value(item, field)
            if item_value is None:
                return False
            
            if min_value is not None and item_value < min_value:
                return False
            
            if max_value is not None and item_value > max_value:
                return False
            
            return True
        
        self.add_filter(range_filter)
    
    def filter_items(self, items: List[Any]) -> List[Any]:
        """Filtre une liste d'éléments"""
        filtered = items
        
        for filter_func in self.filters:
            filtered = [item for item in filtered if filter_func(item)]
        
        return filtered
    
    def _get_nested_value(self, obj: Any, field: str, default: Any = None) -> Any:
        """Récupère une valeur imbriquée"""
        try:
            value = obj
            for part in field.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = getattr(value, part, None)
                
                if value is None:
                    return default
            
            return value
        except (AttributeError, KeyError, TypeError):
            return default
    
    def _apply_operator(self, item_value: Any, filter_value: Any, operator: str) -> bool:
        """Applique un opérateur de comparaison"""
        if item_value is None:
            return False
        
        try:
            if operator == 'eq':
                return item_value == filter_value
            elif operator == 'ne':
                return item_value != filter_value
            elif operator == 'gt':
                return item_value > filter_value
            elif operator == 'gte':
                return item_value >= filter_value
            elif operator == 'lt':
                return item_value < filter_value
            elif operator == 'lte':
                return item_value <= filter_value
            elif operator == 'in':
                return item_value in filter_value
            elif operator == 'nin':
                return item_value not in filter_value
            elif operator == 'contains':
                return filter_value in str(item_value)
            elif operator == 'startswith':
                return str(item_value).startswith(str(filter_value))
            elif operator == 'endswith':
                return str(item_value).endswith(str(filter_value))
            else:
                return False
        except (TypeError, ValueError):
            return False


class Aggregator:
    """Agrégateur de données"""
    
    @staticmethod
    def group_by(items: List[Dict[str, Any]], field: str) -> Dict[Any, List[Dict[str, Any]]]:
        """Groupe les éléments par champ"""
        groups = defaultdict(list)
        
        for item in items:
            key = Aggregator._get_nested_value(item, field)
            groups[key].append(item)
        
        return dict(groups)
    
    @staticmethod
    def count_by(items: List[Dict[str, Any]], field: str) -> Dict[Any, int]:
        """Compte les éléments par champ"""
        counter = Counter()
        
        for item in items:
            key = Aggregator._get_nested_value(item, field)
            counter[key] += 1
        
        return dict(counter)
    
    @staticmethod
    def sum_by(items: List[Dict[str, Any]], group_field: str, sum_field: str) -> Dict[Any, float]:
        """Somme par groupe"""
        groups = Aggregator.group_by(items, group_field)
        result = {}
        
        for key, group_items in groups.items():
            total = sum(
                float(Aggregator._get_nested_value(item, sum_field, 0))
                for item in group_items
            )
            result[key] = total
        
        return result
    
    @staticmethod
    def avg_by(items: List[Dict[str, Any]], group_field: str, avg_field: str) -> Dict[Any, float]:
        """Moyenne par groupe"""
        groups = Aggregator.group_by(items, group_field)
        result = {}
        
        for key, group_items in groups.items():
            values = [
                float(Aggregator._get_nested_value(item, avg_field, 0))
                for item in group_items
            ]
            result[key] = sum(values) / len(values) if values else 0
        
        return result
    
    @staticmethod
    def _get_nested_value(obj: Dict[str, Any], field: str, default: Any = None) -> Any:
        """Récupère une valeur imbriquée"""
        try:
            value = obj
            for part in field.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default


class Serializer:
    """Sérialiseur de données"""
    
    @staticmethod
    def serialize(data: Any, format_type: SerializationFormat = SerializationFormat.JSON,
                  compression: CompressionType = CompressionType.NONE) -> Union[str, bytes]:
        """Sérialise les données"""
        # Conversion en format
        if format_type == SerializationFormat.JSON:
            serialized = json.dumps(data, default=Serializer._json_serializer, indent=2)
        elif format_type == SerializationFormat.YAML:
            try:
                import yaml
                serialized = yaml.dump(data, default_flow_style=False)
            except ImportError:
                serialized = json.dumps(data, default=Serializer._json_serializer)
        elif format_type == SerializationFormat.XML:
            serialized = Serializer._to_xml(data)
        elif format_type == SerializationFormat.PICKLE:
            import pickle
            serialized = pickle.dumps(data)
        else:
            serialized = str(data)
        
        # Compression
        if compression == CompressionType.GZIP:
            import gzip
            return gzip.compress(serialized.encode() if isinstance(serialized, str) else serialized)
        elif compression == CompressionType.ZLIB:
            return zlib.compress(serialized.encode() if isinstance(serialized, str) else serialized)
        elif compression == CompressionType.BROTLI:
            try:
                import brotli
                return brotli.compress(serialized.encode() if isinstance(serialized, str) else serialized)
            except ImportError:
                return serialized
        
        return serialized
    
    @staticmethod
    def deserialize(data: Union[str, bytes], format_type: SerializationFormat = SerializationFormat.JSON,
                    compression: CompressionType = CompressionType.NONE) -> Any:
        """Désérialise les données"""
        # Décompression
        if compression == CompressionType.GZIP:
            import gzip
            data = gzip.decompress(data).decode()
        elif compression == CompressionType.ZLIB:
            data = zlib.decompress(data).decode()
        elif compression == CompressionType.BROTLI:
            try:
                import brotli
                data = brotli.decompress(data).decode()
            except ImportError:
                pass
        
        # Désérialisation
        if format_type == SerializationFormat.JSON:
            return json.loads(data)
        elif format_type == SerializationFormat.YAML:
            try:
                import yaml
                return yaml.safe_load(data)
            except ImportError:
                return json.loads(data)
        elif format_type == SerializationFormat.PICKLE:
            import pickle
            return pickle.loads(data)
        else:
            return data
    
    @staticmethod
    def _json_serializer(obj):
        """Sérialiseur JSON personnalisé"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, BaseModel):
            return obj.dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    @staticmethod
    def _to_xml(data: Any, root_tag: str = "data") -> str:
        """Convertit en XML"""
        def _dict_to_xml(obj, tag="item"):
            if isinstance(obj, dict):
                xml = f"<{tag}>"
                for k, v in obj.items():
                    xml += _dict_to_xml(v, k)
                xml += f"</{tag}>"
                return xml
            elif isinstance(obj, list):
                xml = f"<{tag}>"
                for item in obj:
                    xml += _dict_to_xml(item, "item")
                xml += f"</{tag}>"
                return xml
            else:
                return f"<{tag}>{str(obj)}</{tag}>"
        
        return f'<?xml version="1.0" encoding="UTF-8"?>{_dict_to_xml(data, root_tag)}'


class HashGenerator:
    """Générateur de hachage"""
    
    @staticmethod
    def generate_hash(data: Any, algorithm: str = 'sha256') -> str:
        """Génère un hachage"""
        # Normalisation des données
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)
        
        # Génération du hachage
        if algorithm == 'md5':
            return hashlib.md5(data_str.encode()).hexdigest()
        elif algorithm == 'sha1':
            return hashlib.sha1(data_str.encode()).hexdigest()
        elif algorithm == 'sha256':
            return hashlib.sha256(data_str.encode()).hexdigest()
        elif algorithm == 'sha512':
            return hashlib.sha512(data_str.encode()).hexdigest()
        else:
            return hashlib.sha256(data_str.encode()).hexdigest()
    
    @staticmethod
    def generate_id(prefix: str = "", length: int = 8) -> str:
        """Génère un ID unique"""
        random_part = base64.urlsafe_b64encode(uuid4().bytes).decode()[:length]
        return f"{prefix}{random_part}" if prefix else random_part


class BatchProcessor(Generic[T]):
    """Processeur par lots"""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_batches(self, items: List[T], processor: Callable[[List[T]], Any]) -> List[Any]:
        """Traite les éléments par lots"""
        batches = self._create_batches(items)
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(processor, batch): batch for batch in batches}
            
            for future in as_completed(future_to_batch):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Erreur de traitement: {e}")
        
        return results
    
    async def process_batches_async(self, items: List[T], processor: Callable[[List[T]], Any]) -> List[Any]:
        """Traite les éléments par lots de manière asynchrone"""
        batches = self._create_batches(items)
        
        tasks = [processor(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrer les exceptions
        return [result for result in results if not isinstance(result, Exception)]
    
    def _create_batches(self, items: List[T]) -> List[List[T]]:
        """Crée les lots"""
        batches = []
        for i in range(0, len(items), self.batch_size):
            batches.append(items[i:i + self.batch_size])
        return batches


class ConfigMerger:
    """Fusionneur de configuration"""
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Fusionne plusieurs configurations"""
        result = {}
        
        for config in configs:
            result = ConfigMerger._deep_merge(result, config)
        
        return result
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Fusion profonde de dictionnaires"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigMerger._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


class PerformanceProfiler:
    """Profileur de performance"""
    
    def __init__(self):
        self.metrics: Dict[str, ProcessingMetrics] = {}
    
    def start_profiling(self, operation: str) -> str:
        """Démarre le profilage d'une opération"""
        profile_id = f"{operation}_{uuid4().hex[:8]}"
        self.metrics[profile_id] = ProcessingMetrics(
            start_time=datetime.now(timezone.utc)
        )
        return profile_id
    
    def end_profiling(self, profile_id: str, items_processed: int = 0,
                      errors_count: int = 0, warnings_count: int = 0):
        """Termine le profilage"""
        if profile_id in self.metrics:
            metrics = self.metrics[profile_id]
            metrics.finish()
            metrics.items_processed = items_processed
            metrics.errors_count = errors_count
            metrics.warnings_count = warnings_count
    
    def get_metrics(self, profile_id: str) -> Optional[ProcessingMetrics]:
        """Récupère les métriques"""
        return self.metrics.get(profile_id)
    
    def get_summary(self) -> Dict[str, Any]:
        """Récupère un résumé des métriques"""
        total_operations = len(self.metrics)
        total_duration = sum(
            m.duration_ms or 0 for m in self.metrics.values()
        )
        total_items = sum(m.items_processed for m in self.metrics.values())
        total_errors = sum(m.errors_count for m in self.metrics.values())
        
        return {
            'total_operations': total_operations,
            'total_duration_ms': total_duration,
            'total_items_processed': total_items,
            'total_errors': total_errors,
            'avg_duration_ms': total_duration / total_operations if total_operations > 0 else 0,
            'success_rate': ((total_items - total_errors) / total_items * 100) if total_items > 0 else 0
        }


class RetryManager:
    """Gestionnaire de tentatives"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 exponential_backoff: bool = True, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.exponential_backoff = exponential_backoff
        self.max_delay = max_delay
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute une fonction avec retry"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                delay = self._calculate_delay(attempt)
                asyncio.sleep(delay) if asyncio.iscoroutinefunction(func) else __import__('time').sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calcule le délai d'attente"""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay
        
        return min(delay, self.max_delay)


__all__ = [
    'CacheStrategy', 'SerializationFormat', 'CompressionType', 'ProcessingMetrics',
    'DataTransformer', 'FilterEngine', 'Aggregator', 'Serializer', 'HashGenerator',
    'BatchProcessor', 'ConfigMerger', 'PerformanceProfiler', 'RetryManager'
]
