#!/usr/bin/env python3
"""
Advanced Utils Module for PagerDuty Integration Scripts.

This module provides industrial-grade utilities for PagerDuty integration,
including API clients, security features, caching, monitoring, and more.

Features:
- Enhanced API client with retry logic and circuit breaker
- AES-256 encryption for sensitive data
- Redis-based caching with intelligent TTL
- Rate limiting with exponential backoff
- Comprehensive security and audit logging
- Performance metrics collection
- Health monitoring and alerting
- Data transformation and validation utilities

Author: Enterprise Development Team
Version: 2.0.0
"""

import logging
from typing import Optional

# Import all utility modules
from .api_client import PagerDutyAPIClient, APIError, RateLimitError
from .encryption import SecurityManager, EncryptionError
from .formatters import (
    AlertFormatter, 
    IncidentFormatter, 
    NotificationFormatter,
    JSONFormatter,
    XMLFormatter
)
from .validators import (
    InputValidator, 
    PayloadValidator, 
    SecurityValidator,
    ConfigValidator
)
from .cache_manager import CacheManager, CacheError
from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from .rate_limiter import RateLimiter, RateLimitExceededError
from .metrics_collector import MetricsCollector, PrometheusExporter
from .config_parser import ConfigParser, ConfigError
from .data_transformer import DataTransformer, TransformationError
from .notification_builder import NotificationBuilder, TemplateError
from .webhook_processor import WebhookProcessor, WebhookValidationError
from .audit_logger import AuditLogger, SecurityEvent
from .error_handler import ErrorHandler, UnhandledError
from .health_monitor import HealthMonitor, HealthCheckError

# Configure module-level logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Export all public classes and functions
__all__ = [
    # Core API client
    'PagerDutyAPIClient',
    'APIError',
    'RateLimitError',
    
    # Security utilities
    'SecurityManager',
    'EncryptionError',
    
    # Formatting utilities
    'AlertFormatter',
    'IncidentFormatter', 
    'NotificationFormatter',
    'JSONFormatter',
    'XMLFormatter',
    
    # Validation utilities
    'InputValidator',
    'PayloadValidator',
    'SecurityValidator',
    'ConfigValidator',
    
    # Caching utilities
    'CacheManager',
    'CacheError',
    
    # Resilience utilities
    'CircuitBreaker',
    'CircuitBreakerOpenError',
    'RateLimiter',
    'RateLimitExceededError',
    
    # Monitoring utilities
    'MetricsCollector',
    'PrometheusExporter',
    'HealthMonitor',
    'HealthCheckError',
    
    # Configuration utilities
    'ConfigParser',
    'ConfigError',
    
    # Data processing utilities
    'DataTransformer',
    'TransformationError',
    'NotificationBuilder',
    'TemplateError',
    'WebhookProcessor',
    'WebhookValidationError',
    
    # Logging and error handling
    'AuditLogger',
    'SecurityEvent',
    'ErrorHandler',
    'UnhandledError',
]

# Module metadata
__version__ = "2.0.0"
__author__ = "Enterprise Development Team"
__description__ = "Advanced PagerDuty integration utilities"

# Default configuration
DEFAULT_CONFIG = {
    'api_timeout': 30,
    'retry_attempts': 3,
    'cache_ttl': 300,
    'circuit_breaker_threshold': 5,
    'rate_limit_per_minute': 60,
    'encryption_algorithm': 'AES-256-GCM',
    'log_level': 'INFO'
}

def get_version() -> str:
    """Get the current version of the utils module."""
    return __version__

def configure_logging(level: str = "INFO", 
                     format_string: Optional[str] = None) -> None:
    """Configure logging for the utils module."""
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(funcName)s:%(lineno)d - %(message)s"
        )
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def initialize_utils(config: Optional[dict] = None) -> dict:
    """
    Initialize the utils module with configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        dict: Effective configuration used
    """
    effective_config = DEFAULT_CONFIG.copy()
    if config:
        effective_config.update(config)
    
    # Configure logging
    configure_logging(effective_config.get('log_level', 'INFO'))
    
    logger = logging.getLogger(__name__)
    logger.info(f"Utils module initialized with version {__version__}")
    logger.debug(f"Configuration: {effective_config}")
    
    return effective_config
- Formateurs de messages et données
- Gestionnaire de chiffrement sécurisé
- Client API PagerDuty optimisé
- Utilitaires de logging et monitoring
- Helpers pour la gestion des erreurs

Version: 1.0.0
Auteur: Spotify AI Agent Team
"""

import re
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Version et metadata du module
__version__ = "1.0.0"
__author__ = "Spotify AI Agent Team"

# Constantes globales
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PHONE_REGEX = re.compile(r'^\+?1?-?\.?\s?\(?(\d{3})\)?[-\.\s]?(\d{3})[-\.\s]?(\d{4})$')
UUID_REGEX = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.IGNORECASE)

# Types de données supportés
SUPPORTED_DATA_TYPES = {
    "string", "integer", "float", "boolean", "list", "dict", "datetime"
}

# Niveaux de sévérité standard
SEVERITY_LEVELS = {
    "critical": 1,
    "high": 2, 
    "medium": 3,
    "low": 4,
    "info": 5
}

def get_version() -> str:
    """Retourne la version du module utilities."""
    return __version__

def get_timestamp(format_iso: bool = True) -> Union[str, datetime]:
    """
    Retourne un timestamp actuel.
    
    Args:
        format_iso: Si True, retourne au format ISO, sinon objet datetime
        
    Returns:
        Timestamp au format spécifié
    """
    now = datetime.now(timezone.utc)
    return now.isoformat() if format_iso else now

def safe_json_loads(data: str, default: Any = None) -> Any:
    """
    Parse JSON de manière sécurisée.
    
    Args:
        data: Chaîne JSON à parser
        default: Valeur par défaut en cas d'erreur
        
    Returns:
        Données parsées ou valeur par défaut
    """
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(data: Any, default: Any = None, **kwargs) -> Optional[str]:
    """
    Sérialise en JSON de manière sécurisée.
    
    Args:
        data: Données à sérialiser
        default: Valeur par défaut en cas d'erreur
        **kwargs: Arguments pour json.dumps
        
    Returns:
        Chaîne JSON ou None en cas d'erreur
    """
    try:
        return json.dumps(data, default=str, ensure_ascii=False, **kwargs)
    except (TypeError, ValueError):
        return default

def generate_hash(data: str, algorithm: str = "sha256") -> str:
    """
    Génère un hash pour les données fournies.
    
    Args:
        data: Données à hasher
        algorithm: Algorithme de hash (md5, sha1, sha256, sha512)
        
    Returns:
        Hash hexadécimal
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data.encode('utf-8'))
    return hash_obj.hexdigest()

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Tronque une chaîne si elle dépasse la longueur maximale.
    
    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
        suffix: Suffixe à ajouter si tronqué
        
    Returns:
        Texte tronqué si nécessaire
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def deep_merge_dicts(dict1: Dict[Any, Any], dict2: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Fusionne récursivement deux dictionnaires.
    
    Args:
        dict1: Premier dictionnaire
        dict2: Deuxième dictionnaire
        
    Returns:
        Dictionnaire fusionné
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def flatten_dict(data: Dict[Any, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Aplatis un dictionnaire imbriqué.
    
    Args:
        data: Dictionnaire à aplatir
        parent_key: Clé parent pour la récursion
        sep: Séparateur pour les clés
        
    Returns:
        Dictionnaire aplati
    """
    items = []
    
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    
    return dict(items)

def sanitize_filename(filename: str) -> str:
    """
    Nettoie un nom de fichier en supprimant les caractères invalides.
    
    Args:
        filename: Nom de fichier à nettoyer
        
    Returns:
        Nom de fichier nettoyé
    """
    # Supprimer les caractères non autorisés
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Supprimer les espaces en début/fin et les points
    filename = filename.strip(' .')
    
    # Limiter la longueur
    return filename[:255] if len(filename) > 255 else filename

def extract_numbers(text: str) -> List[Union[int, float]]:
    """
    Extrait tous les nombres d'un texte.
    
    Args:
        text: Texte à analyser
        
    Returns:
        Liste des nombres trouvés
    """
    numbers = []
    
    # Recherche des nombres entiers et décimaux
    number_pattern = re.compile(r'-?\d+\.?\d*')
    matches = number_pattern.findall(text)
    
    for match in matches:
        try:
            if '.' in match:
                numbers.append(float(match))
            else:
                numbers.append(int(match))
        except ValueError:
            continue
    
    return numbers

def format_bytes(bytes_value: int) -> str:
    """
    Formate une taille en bytes en format lisible.
    
    Args:
        bytes_value: Taille en bytes
        
    Returns:
        Taille formatée (ex: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} EB"

def format_duration(seconds: float) -> str:
    """
    Formate une durée en secondes en format lisible.
    
    Args:
        seconds: Durée en secondes
        
    Returns:
        Durée formatée (ex: "2h 30m 15s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    
    if hours < 24:
        return f"{hours}h {remaining_minutes}m"
    
    days = int(hours // 24)
    remaining_hours = hours % 24
    
    return f"{days}d {remaining_hours}h"

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Divise une liste en chunks de taille spécifiée.
    
    Args:
        lst: Liste à diviser
        chunk_size: Taille de chaque chunk
        
    Returns:
        Liste de chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def retry_on_exception(
    exceptions: Union[Exception, tuple] = Exception,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """
    Décorateur pour retry automatique en cas d'exception.
    
    Args:
        exceptions: Exception(s) à capturer
        max_retries: Nombre maximum de tentatives
        delay: Délai initial entre les tentatives
        backoff: Facteur de multiplication du délai
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise e
                    
                    import time
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator

class CacheManager:
    """Gestionnaire de cache simple en mémoire"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.now(timezone.utc) < expiry:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stocke une valeur dans le cache"""
        ttl = ttl or self.default_ttl
        expiry = datetime.now(timezone.utc).timestamp() + ttl
        self.cache[key] = (value, datetime.fromtimestamp(expiry, timezone.utc))
    
    def delete(self, key: str) -> bool:
        """Supprime une valeur du cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Vide le cache"""
        self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """Nettoie les entrées expirées"""
        now = datetime.now(timezone.utc)
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if now >= expiry
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)

class RateLimiter:
    """Limiteur de taux simple"""
    
    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls = []
    
    def can_proceed(self) -> bool:
        """Vérifie si un appel peut être effectué"""
        now = datetime.now(timezone.utc).timestamp()
        
        # Nettoyer les anciens appels
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < self.window_seconds]
        
        return len(self.calls) < self.max_calls
    
    def record_call(self) -> None:
        """Enregistre un appel"""
        self.calls.append(datetime.now(timezone.utc).timestamp())
    
    def wait_time(self) -> float:
        """Retourne le temps d'attente nécessaire"""
        if self.can_proceed():
            return 0.0
        
        now = datetime.now(timezone.utc).timestamp()
        oldest_call = min(self.calls)
        return (oldest_call + self.window_seconds) - now

# Cache global pour le module
_global_cache = CacheManager()

def get_cache() -> CacheManager:
    """Retourne l'instance de cache global"""
    return _global_cache

# Export des symboles publics
__all__ = [
    "get_version",
    "get_timestamp", 
    "safe_json_loads",
    "safe_json_dumps",
    "generate_hash",
    "truncate_string",
    "deep_merge_dicts",
    "flatten_dict",
    "sanitize_filename",
    "extract_numbers",
    "format_bytes",
    "format_duration",
    "chunk_list",
    "retry_on_exception",
    "CacheManager",
    "RateLimiter",
    "get_cache",
    "EMAIL_REGEX",
    "PHONE_REGEX", 
    "UUID_REGEX",
    "SEVERITY_LEVELS"
]
