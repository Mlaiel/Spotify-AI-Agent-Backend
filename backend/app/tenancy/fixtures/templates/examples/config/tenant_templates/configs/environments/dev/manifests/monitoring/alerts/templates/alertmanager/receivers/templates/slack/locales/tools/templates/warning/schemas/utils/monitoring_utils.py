"""
Monitoring Utils - Utilitaires pour le Système de Monitoring
===========================================================

Fonctions utilitaires, helpers et outils communs pour le système
de monitoring ultra-avancé Spotify AI Agent.

Features:
    - Formatage et validation des données
    - Utilitaires de calcul et statistiques
    - Helpers de configuration et logging
    - Outils de performance et optimisation
    - Convertisseurs et transformateurs de données

Author: Expert DevOps Utils Team + Senior Backend Engineer + System Architect
"""

import asyncio
import json
import yaml
import csv
import gzip
import pickle
import hashlib
import base64
import time
import re
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, is_dataclass
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict, deque, OrderedDict
from contextlib import contextmanager
from functools import wraps, lru_cache
from itertools import islice
import logging
import warnings
import sys
import os
import tempfile
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref

# Imports conditionnels
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION ET CONSTANTES
# =============================================================================

# Constantes pour les types de métriques
METRIC_TYPES = {
    'GAUGE': 'gauge',
    'COUNTER': 'counter',
    'HISTOGRAM': 'histogram',
    'SUMMARY': 'summary'
}

# Niveaux de sévérité
SEVERITY_LEVELS = {
    'CRITICAL': 1,
    'HIGH': 2,
    'MEDIUM': 3,
    'LOW': 4,
    'INFO': 5
}

# Formats de date supportés
DATE_FORMATS = [
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%dT%H:%M:%S.%f',
    '%Y-%m-%dT%H:%M:%S.%fZ',
    '%Y-%m-%d',
    '%H:%M:%S'
]

# Unités de mesure
UNITS = {
    'bytes': ['B', 'KB', 'MB', 'GB', 'TB', 'PB'],
    'time': ['ns', 'μs', 'ms', 's', 'm', 'h', 'd'],
    'rate': ['ops/s', 'req/s', 'msg/s', 'pkt/s'],
    'percentage': ['%'],
    'count': ['count', 'items', 'events']
}


# =============================================================================
# UTILITAIRES DE DONNÉES
# =============================================================================

@dataclass
class DataPoint:
    """Point de données avec métadonnées."""
    timestamp: datetime
    value: float
    metric_name: str
    tenant_id: str
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}


class DataValidator:
    """Validateur de données pour le monitoring."""
    
    @staticmethod
    def validate_metric_name(name: str) -> bool:
        """Valide un nom de métrique."""
        if not isinstance(name, str):
            return False
        
        # Pattern: lettres, chiffres, underscore, point, tiret
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_.-]*$'
        return bool(re.match(pattern, name)) and len(name) <= 255
    
    @staticmethod
    def validate_tenant_id(tenant_id: str) -> bool:
        """Valide un ID de tenant."""
        if not isinstance(tenant_id, str):
            return False
        
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, tenant_id)) and len(tenant_id) <= 100
    
    @staticmethod
    def validate_timestamp(timestamp: Union[datetime, str, int, float]) -> Optional[datetime]:
        """Valide et convertit un timestamp."""
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, str):
            for fmt in DATE_FORMATS:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
            return None
        
        if isinstance(timestamp, (int, float)):
            try:
                # Assumons timestamp Unix
                if timestamp > 1e10:  # Timestamp en millisecondes
                    timestamp = timestamp / 1000
                return datetime.fromtimestamp(timestamp, timezone.utc)
            except (ValueError, OSError):
                return None
        
        return None
    
    @staticmethod
    def validate_numeric_value(value: Any) -> Optional[float]:
        """Valide et convertit une valeur numérique."""
        if isinstance(value, (int, float)):
            if not math.isnan(value) and math.isfinite(value):
                return float(value)
        
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        
        return None
    
    @staticmethod
    def validate_tags(tags: Dict[str, str]) -> Dict[str, str]:
        """Valide et nettoie les tags."""
        if not isinstance(tags, dict):
            return {}
        
        validated_tags = {}
        for key, value in tags.items():
            if isinstance(key, str) and isinstance(value, str):
                # Nettoyer les clés et valeurs
                clean_key = re.sub(r'[^a-zA-Z0-9_-]', '_', key)[:50]
                clean_value = str(value)[:100]
                validated_tags[clean_key] = clean_value
        
        return validated_tags


class DataTransformer:
    """Transformateur de données pour différents formats."""
    
    @staticmethod
    def to_dataframe(data_points: List[DataPoint], include_metadata: bool = False):
        """Convertit une liste de DataPoints en DataFrame pandas."""
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas requis pour cette fonctionnalité")
        
        if not data_points:
            return pd.DataFrame()
        
        rows = []
        for dp in data_points:
            row = {
                'timestamp': dp.timestamp,
                'value': dp.value,
                'metric_name': dp.metric_name,
                'tenant_id': dp.tenant_id
            }
            
            # Ajouter les tags comme colonnes
            for tag_key, tag_value in dp.tags.items():
                row[f'tag_{tag_key}'] = tag_value
            
            if include_metadata:
                row['metadata'] = dp.metadata
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def from_dataframe(df, timestamp_col: str = 'timestamp', 
                      value_col: str = 'value') -> List[DataPoint]:
        """Convertit un DataFrame en liste de DataPoints."""
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas requis pour cette fonctionnalité")
        
        data_points = []
        
        for _, row in df.iterrows():
            # Extraire les colonnes de base
            timestamp = DataValidator.validate_timestamp(row[timestamp_col])
            value = DataValidator.validate_numeric_value(row[value_col])
            
            if timestamp is None or value is None:
                continue
            
            # Extraire les tags
            tags = {}
            for col in df.columns:
                if col.startswith('tag_'):
                    tag_key = col[4:]  # Enlever 'tag_'
                    tags[tag_key] = str(row[col])
            
            # Créer le DataPoint
            dp = DataPoint(
                timestamp=timestamp,
                value=value,
                metric_name=row.get('metric_name', 'unknown'),
                tenant_id=row.get('tenant_id', 'unknown'),
                tags=tags,
                metadata=row.get('metadata', {})
            )
            
            data_points.append(dp)
        
        return data_points
    
    @staticmethod
    def to_json(data_points: List[DataPoint], 
                ensure_ascii: bool = False) -> str:
        """Convertit une liste de DataPoints en JSON."""
        serializable_data = []
        
        for dp in data_points:
            item = {
                'timestamp': dp.timestamp.isoformat(),
                'value': dp.value,
                'metric_name': dp.metric_name,
                'tenant_id': dp.tenant_id,
                'tags': dp.tags,
                'metadata': dp.metadata
            }
            serializable_data.append(item)
        
        return json.dumps(serializable_data, ensure_ascii=ensure_ascii, indent=2)
    
    @staticmethod
    def from_json(json_str: str) -> List[DataPoint]:
        """Convertit du JSON en liste de DataPoints."""
        try:
            data = json.loads(json_str)
            data_points = []
            
            for item in data:
                timestamp = DataValidator.validate_timestamp(item['timestamp'])
                value = DataValidator.validate_numeric_value(item['value'])
                
                if timestamp and value is not None:
                    dp = DataPoint(
                        timestamp=timestamp,
                        value=value,
                        metric_name=item.get('metric_name', 'unknown'),
                        tenant_id=item.get('tenant_id', 'unknown'),
                        tags=item.get('tags', {}),
                        metadata=item.get('metadata', {})
                    )
                    data_points.append(dp)
            
            return data_points
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Erreur parsing JSON: {e}")
            return []
    
    @staticmethod
    def to_csv(data_points: List[DataPoint], output_path: str = None) -> Optional[str]:
        """Convertit une liste de DataPoints en CSV."""
        if not data_points:
            return None
        
        output = output_path or tempfile.mktemp(suffix='.csv')
        
        try:
            with open(output, 'w', newline='', encoding='utf-8') as csvfile:
                # Déterminer toutes les colonnes de tags
                all_tags = set()
                for dp in data_points:
                    all_tags.update(dp.tags.keys())
                
                fieldnames = ['timestamp', 'value', 'metric_name', 'tenant_id']
                fieldnames.extend(f'tag_{tag}' for tag in sorted(all_tags))
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for dp in data_points:
                    row = {
                        'timestamp': dp.timestamp.isoformat(),
                        'value': dp.value,
                        'metric_name': dp.metric_name,
                        'tenant_id': dp.tenant_id
                    }
                    
                    # Ajouter les tags
                    for tag in all_tags:
                        row[f'tag_{tag}'] = dp.tags.get(tag, '')
                    
                    writer.writerow(row)
            
            return output
            
        except Exception as e:
            logger.error(f"Erreur écriture CSV: {e}")
            return None


# =============================================================================
# UTILITAIRES DE CALCUL ET STATISTIQUES
# =============================================================================

class StatisticsCalculator:
    """Calculateur de statistiques pour les métriques."""
    
    @staticmethod
    def basic_stats(values: List[float]) -> Dict[str, float]:
        """Calcule les statistiques de base."""
        if not values:
            return {}
        
        import statistics
        import math
        
        sorted_values = sorted(values)
        n = len(values)
        
        stats = {
            'count': n,
            'sum': sum(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'range': max(values) - min(values)
        }
        
        if n > 1:
            stats['std'] = statistics.stdev(values)
            stats['variance'] = statistics.variance(values)
        else:
            stats['std'] = 0.0
            stats['variance'] = 0.0
        
        # Percentiles
        stats['p25'] = StatisticsCalculator.percentile(sorted_values, 25)
        stats['p75'] = StatisticsCalculator.percentile(sorted_values, 75)
        stats['p90'] = StatisticsCalculator.percentile(sorted_values, 90)
        stats['p95'] = StatisticsCalculator.percentile(sorted_values, 95)
        stats['p99'] = StatisticsCalculator.percentile(sorted_values, 99)
        
        # IQR
        stats['iqr'] = stats['p75'] - stats['p25']
        
        return stats
    
    @staticmethod
    def percentile(sorted_values: List[float], percent: float) -> float:
        """Calcule un percentile."""
        if not sorted_values:
            return 0.0
        
        k = (len(sorted_values) - 1) * (percent / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_values[int(k)]
        
        d0 = sorted_values[int(f)] * (c - k)
        d1 = sorted_values[int(c)] * (k - f)
        
        return d0 + d1
    
    @staticmethod
    def moving_average(values: List[float], window: int) -> List[float]:
        """Calcule une moyenne mobile."""
        if window <= 0 or len(values) < window:
            return values.copy()
        
        result = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx:i + 1]
            result.append(sum(window_values) / len(window_values))
        
        return result
    
    @staticmethod
    def exponential_moving_average(values: List[float], alpha: float = 0.1) -> List[float]:
        """Calcule une moyenne mobile exponentielle."""
        if not values:
            return []
        
        result = [values[0]]
        
        for i in range(1, len(values)):
            ema = alpha * values[i] + (1 - alpha) * result[-1]
            result.append(ema)
        
        return result
    
    @staticmethod
    def detect_outliers_iqr(values: List[float], factor: float = 1.5) -> List[int]:
        """Détecte les outliers avec la méthode IQR."""
        if len(values) < 4:
            return []
        
        sorted_values = sorted(values)
        q1 = StatisticsCalculator.percentile(sorted_values, 25)
        q3 = StatisticsCalculator.percentile(sorted_values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        outliers = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)
        
        return outliers
    
    @staticmethod
    def detect_outliers_zscore(values: List[float], threshold: float = 3.0) -> List[int]:
        """Détecte les outliers avec le Z-score."""
        if len(values) < 2:
            return []
        
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        
        if std == 0:
            return []
        
        outliers = []
        for i, value in enumerate(values):
            z_score = abs((value - mean) / std)
            if z_score > threshold:
                outliers.append(i)
        
        return outliers
    
    @staticmethod
    def correlation(values1: List[float], values2: List[float]) -> float:
        """Calcule la corrélation de Pearson."""
        if len(values1) != len(values2) or len(values1) < 2:
            return 0.0
        
        n = len(values1)
        sum1 = sum(values1)
        sum2 = sum(values2)
        sum1_sq = sum(x * x for x in values1)
        sum2_sq = sum(x * x for x in values2)
        sum_products = sum(x * y for x, y in zip(values1, values2))
        
        numerator = n * sum_products - sum1 * sum2
        denominator = math.sqrt((n * sum1_sq - sum1 * sum1) * (n * sum2_sq - sum2 * sum2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


# =============================================================================
# UTILITAIRES DE FORMATAGE
# =============================================================================

class Formatter:
    """Utilitaires de formatage pour l'affichage."""
    
    @staticmethod
    def format_bytes(bytes_value: float, decimal_places: int = 2) -> str:
        """Formate une valeur en bytes avec les unités appropriées."""
        if bytes_value == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        unit_index = 0
        
        while bytes_value >= 1024 and unit_index < len(units) - 1:
            bytes_value /= 1024
            unit_index += 1
        
        return f"{bytes_value:.{decimal_places}f} {units[unit_index]}"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Formate une durée en secondes."""
        if seconds < 0:
            return "0s"
        
        units = [
            ('d', 86400),
            ('h', 3600),
            ('m', 60),
            ('s', 1)
        ]
        
        parts = []
        remaining = int(seconds)
        
        for unit_name, unit_seconds in units:
            if remaining >= unit_seconds:
                count = remaining // unit_seconds
                remaining = remaining % unit_seconds
                parts.append(f"{count}{unit_name}")
        
        if not parts:
            if seconds < 1:
                return f"{seconds*1000:.0f}ms"
            else:
                return f"{seconds:.1f}s"
        
        return " ".join(parts)
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 1) -> str:
        """Formate un pourcentage."""
        return f"{value:.{decimal_places}f}%"
    
    @staticmethod
    def format_number(value: float, decimal_places: int = 2, 
                     use_thousands_separator: bool = True) -> str:
        """Formate un nombre avec séparateurs de milliers."""
        formatted = f"{value:.{decimal_places}f}"
        
        if use_thousands_separator:
            # Séparer la partie entière et décimale
            if '.' in formatted:
                integer_part, decimal_part = formatted.split('.')
                integer_part = f"{int(integer_part):,}"
                formatted = f"{integer_part}.{decimal_part}"
            else:
                formatted = f"{int(value):,}"
        
        return formatted
    
    @staticmethod
    def format_rate(value: float, unit: str = "ops/s", decimal_places: int = 2) -> str:
        """Formate un taux."""
        if value >= 1000000:
            return f"{value/1000000:.{decimal_places}f}M {unit}"
        elif value >= 1000:
            return f"{value/1000:.{decimal_places}f}K {unit}"
        else:
            return f"{value:.{decimal_places}f} {unit}"
    
    @staticmethod
    def format_timestamp(timestamp: datetime, 
                        format_type: str = "iso") -> str:
        """Formate un timestamp."""
        if format_type == "iso":
            return timestamp.isoformat()
        elif format_type == "human":
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")
        elif format_type == "compact":
            return timestamp.strftime("%m/%d %H:%M")
        elif format_type == "relative":
            return Formatter.format_relative_time(timestamp)
        else:
            return timestamp.strftime(format_type)
    
    @staticmethod
    def format_relative_time(timestamp: datetime) -> str:
        """Formate un timestamp en temps relatif."""
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        delta = now - timestamp
        seconds = delta.total_seconds()
        
        if seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            return f"{int(seconds//60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds//3600)}h ago"
        elif seconds < 2592000:  # 30 days
            return f"{int(seconds//86400)}d ago"
        else:
            return timestamp.strftime("%Y-%m-%d")


# =============================================================================
# UTILITAIRES DE CACHE ET PERFORMANCE
# =============================================================================

class CacheManager:
    """Gestionnaire de cache pour améliorer les performances."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Any:
        """Récupère une valeur du cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Vérifier l'expiration
            if self._is_expired(key):
                self._remove(key)
                return None
            
            # Déplacer en fin (LRU)
            value = self._cache.pop(key)
            self._cache[key] = value
            
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Stocke une valeur dans le cache."""
        with self._lock:
            # Supprimer l'ancienne valeur si elle existe
            if key in self._cache:
                self._remove(key)
            
            # Faire de la place si nécessaire
            while len(self._cache) >= self.max_size:
                self._remove_oldest()
            
            # Ajouter la nouvelle valeur
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """Supprime une valeur du cache."""
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False
    
    def clear(self) -> None:
        """Vide le cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def size(self) -> int:
        """Retourne la taille actuelle du cache."""
        return len(self._cache)
    
    def _is_expired(self, key: str) -> bool:
        """Vérifie si une entrée est expirée."""
        if key not in self._timestamps:
            return True
        
        age = time.time() - self._timestamps[key]
        return age > self.ttl_seconds
    
    def _remove(self, key: str) -> None:
        """Supprime une entrée du cache."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def _remove_oldest(self) -> None:
        """Supprime l'entrée la plus ancienne."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)


def cached(ttl_seconds: int = 3600, max_size: int = 100):
    """Décorateur pour mettre en cache les résultats de fonction."""
    cache = CacheManager(max_size=max_size, ttl_seconds=ttl_seconds)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Créer une clé de cache
            cache_key = _create_cache_key(func.__name__, args, kwargs)
            
            # Vérifier le cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Calculer le résultat
            result = func(*args, **kwargs)
            
            # Mettre en cache
            cache.set(cache_key, result)
            
            return result
        
        # Ajouter des méthodes pour gérer le cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_size = cache.size
        
        return wrapper
    
    return decorator


def _create_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Crée une clé de cache unique."""
    key_data = {
        'func': func_name,
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


# =============================================================================
# UTILITAIRES DE CONFIGURATION
# =============================================================================

class ConfigManager:
    """Gestionnaire de configuration avec support multi-format."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        self.watchers = []
        
        if config_path:
            self.load()
    
    def load(self, config_path: Optional[str] = None) -> bool:
        """Charge la configuration depuis un fichier."""
        path = config_path or self.config_path
        if not path or not os.path.exists(path):
            return False
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    self.config = yaml.safe_load(f) or {}
                elif path.endswith('.json'):
                    self.config = json.load(f)
                else:
                    logger.warning(f"Format de fichier non supporté: {path}")
                    return False
            
            logger.info(f"Configuration chargée depuis {path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement configuration: {e}")
            return False
    
    def save(self, config_path: Optional[str] = None) -> bool:
        """Sauvegarde la configuration dans un fichier."""
        path = config_path or self.config_path
        if not path:
            return False
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    yaml.safe_dump(self.config, f, default_flow_style=False)
                elif path.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    logger.warning(f"Format de fichier non supporté: {path}")
                    return False
            
            logger.info(f"Configuration sauvegardée dans {path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de configuration avec support des clés imbriquées."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Définit une valeur de configuration avec support des clés imbriquées."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Met à jour la configuration avec un dictionnaire."""
        self._deep_update(self.config, updates)
    
    def _deep_update(self, target: dict, source: dict) -> None:
        """Mise à jour récursive de dictionnaire."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value


# =============================================================================
# UTILITAIRES DE LOGGING
# =============================================================================

class MonitoringLogger:
    """Logger spécialisé pour le monitoring avec formatage structuré."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Configure les handlers de logging."""
        # Handler console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Handler fichier
        file_handler = logging.FileHandler('monitoring.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Formateurs
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        console_handler.setFormatter(console_format)
        file_handler.setFormatter(file_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def log_metric(self, metric_name: str, value: float, tenant_id: str, 
                   tags: Dict[str, str] = None):
        """Log une métrique de manière structurée."""
        log_data = {
            'type': 'metric',
            'metric_name': metric_name,
            'value': value,
            'tenant_id': tenant_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if tags:
            log_data['tags'] = tags
        
        self.logger.info(json.dumps(log_data))
    
    def log_event(self, event_type: str, description: str, 
                  tenant_id: str = None, metadata: Dict[str, Any] = None):
        """Log un événement de manière structurée."""
        log_data = {
            'type': 'event',
            'event_type': event_type,
            'description': description,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if tenant_id:
            log_data['tenant_id'] = tenant_id
        
        if metadata:
            log_data['metadata'] = metadata
        
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log une erreur avec contexte."""
        log_data = {
            'type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if context:
            log_data['context'] = context
        
        self.logger.error(json.dumps(log_data))


# =============================================================================
# UTILITAIRES DE SYSTÈME
# =============================================================================

class SystemUtils:
    """Utilitaires système pour le monitoring."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Récupère les informations système."""
        info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'pid': os.getpid(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if PSUTIL_AVAILABLE:
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_total': psutil.disk_usage('/').total if os.path.exists('/') else 0,
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
            })
        
        return info
    
    @staticmethod
    def check_disk_space(path: str = '/') -> Dict[str, Any]:
        """Vérifie l'espace disque disponible."""
        if PSUTIL_AVAILABLE:
            usage = psutil.disk_usage(path)
            return {
                'total': usage.total,
                'used': usage.used,
                'free': usage.free,
                'percent': (usage.used / usage.total) * 100
            }
        else:
            return {'error': 'psutil non disponible'}
    
    @staticmethod
    def check_memory_usage() -> Dict[str, Any]:
        """Vérifie l'utilisation mémoire."""
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'used': memory.used,
                'available': memory.available,
                'percent': memory.percent
            }
        else:
            return {'error': 'psutil non disponible'}
    
    @staticmethod
    def get_network_stats() -> Dict[str, Any]:
        """Récupère les statistiques réseau."""
        if PSUTIL_AVAILABLE:
            stats = psutil.net_io_counters()
            return {
                'bytes_sent': stats.bytes_sent,
                'bytes_recv': stats.bytes_recv,
                'packets_sent': stats.packets_sent,
                'packets_recv': stats.packets_recv,
                'errin': stats.errin,
                'errout': stats.errout,
                'dropin': stats.dropin,
                'dropout': stats.dropout
            }
        else:
            return {'error': 'psutil non disponible'}


# =============================================================================
# UTILITAIRES DE SÉCURITÉ
# =============================================================================

class SecurityUtils:
    """Utilitaires de sécurité pour le monitoring."""
    
    @staticmethod
    def hash_sensitive_data(data: str, salt: str = None) -> str:
        """Hash des données sensibles avec salt."""
        if salt is None:
            salt = os.urandom(32).hex()
        
        hash_obj = hashlib.sha256()
        hash_obj.update((data + salt).encode('utf-8'))
        
        return f"{salt}:{hash_obj.hexdigest()}"
    
    @staticmethod
    def verify_hash(data: str, hashed: str) -> bool:
        """Vérifie un hash avec salt."""
        try:
            salt, hash_value = hashed.split(':', 1)
            
            hash_obj = hashlib.sha256()
            hash_obj.update((data + salt).encode('utf-8'))
            
            return hash_obj.hexdigest() == hash_value
        except ValueError:
            return False
    
    @staticmethod
    def sanitize_input(data: str, max_length: int = 1000) -> str:
        """Sanitise les entrées utilisateur."""
        if not isinstance(data, str):
            data = str(data)
        
        # Limiter la longueur
        data = data[:max_length]
        
        # Supprimer les caractères dangereux
        data = re.sub(r'[<>\"\'&]', '', data)
        
        return data.strip()
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Génère un token aléatoire sécurisé."""
        return base64.urlsafe_b64encode(os.urandom(length)).decode('utf-8')[:length]
    
    @staticmethod
    def mask_sensitive_data(data: str, mask_char: str = '*') -> str:
        """Masque les données sensibles pour les logs."""
        if len(data) <= 4:
            return mask_char * len(data)
        
        visible_chars = 2
        masked_chars = len(data) - (visible_chars * 2)
        
        return data[:visible_chars] + (mask_char * masked_chars) + data[-visible_chars:]


# =============================================================================
# TESTS ET UTILITAIRES DE DÉVELOPPEMENT
# =============================================================================

def create_test_data(count: int = 100, 
                    metric_name: str = "test_metric",
                    tenant_id: str = "test_tenant") -> List[DataPoint]:
    """Crée des données de test."""
    import random
    import math
    
    data_points = []
    base_time = datetime.utcnow() - timedelta(hours=count)
    
    for i in range(count):
        timestamp = base_time + timedelta(hours=i)
        
        # Valeur avec pattern et bruit
        value = 50 + 20 * math.sin(2 * math.pi * i / 24) + random.gauss(0, 5)
        value = max(0, value)  # Pas de valeurs négatives
        
        tags = {
            'environment': random.choice(['prod', 'staging', 'dev']),
            'region': random.choice(['us-east', 'us-west', 'eu-west'])
        }
        
        metadata = {
            'source': 'test_generator',
            'index': i
        }
        
        dp = DataPoint(
            timestamp=timestamp,
            value=value,
            metric_name=metric_name,
            tenant_id=tenant_id,
            tags=tags,
            metadata=metadata
        )
        
        data_points.append(dp)
    
    return data_points


@contextmanager
def timer(description: str = "Operation"):
    """Context manager pour mesurer le temps d'exécution."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{description} completed in {duration:.3f} seconds")


def profile_function(func):
    """Décorateur pour profiler une fonction."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            pr.disable()
            
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            
            logger.debug(f"Profile for {func.__name__}:\n{s.getvalue()}")
        
        return result
    
    return wrapper


# =============================================================================
# FONCTIONS UTILITAIRES GLOBALES
# =============================================================================

def deep_merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Fusion récursive de dictionnaires."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Aplatit un dictionnaire imbriqué."""
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Divise une liste en chunks de taille donnée."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Décorateur pour retry automatique avec backoff exponentiel."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    
                    logger.warning(f"Attempt {attempts} failed for {func.__name__}: {e}. "
                                 f"Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise Exception(f"All {max_attempts} attempts failed for {func.__name__}")
        
        return wrapper
    
    return decorator


# =============================================================================
# TESTS
# =============================================================================

def run_utils_tests():
    """Lance les tests des utilitaires."""
    print("=== Tests des Utilitaires de Monitoring ===\n")
    
    # Test DataValidator
    print("1. Test DataValidator...")
    assert DataValidator.validate_metric_name("cpu.usage.percent") == True
    assert DataValidator.validate_metric_name("123invalid") == False
    assert DataValidator.validate_tenant_id("tenant_001") == True
    print("✓ DataValidator OK\n")
    
    # Test StatisticsCalculator
    print("2. Test StatisticsCalculator...")
    values = [1, 2, 3, 4, 5, 10, 100]
    stats = StatisticsCalculator.basic_stats(values)
    assert stats['count'] == 7
    assert stats['median'] == 4
    print("✓ StatisticsCalculator OK\n")
    
    # Test Formatter
    print("3. Test Formatter...")
    assert Formatter.format_bytes(1024) == "1.00 KB"
    assert Formatter.format_duration(3661) == "1h 1m 1s"
    print("✓ Formatter OK\n")
    
    # Test CacheManager
    print("4. Test CacheManager...")
    cache = CacheManager(max_size=2, ttl_seconds=1)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    time.sleep(1.1)
    assert cache.get("key1") is None  # Expiré
    print("✓ CacheManager OK\n")
    
    # Test utilitaires de données
    print("5. Test données...")
    test_data = create_test_data(10)
    assert len(test_data) == 10
    assert all(isinstance(dp, DataPoint) for dp in test_data)
    print("✓ Données OK\n")
    
    print("🎉 Tous les tests passent avec succès!")


if __name__ == "__main__":
    run_utils_tests()
