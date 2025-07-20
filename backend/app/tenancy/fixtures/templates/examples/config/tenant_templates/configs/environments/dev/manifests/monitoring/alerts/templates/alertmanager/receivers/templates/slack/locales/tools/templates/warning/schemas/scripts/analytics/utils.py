"""
Utils Module - Utilitaires Analytics
====================================

Ce module fournit des utilitaires essentiels pour le système d'analytics,
incluant le logging, la validation, le formatage, le timing et le rate limiting.

Classes:
- Logger: Système de logging avancé
- Validator: Validation de données
- Formatter: Formatage de données
- Timer: Mesure de temps
- RateLimiter: Limitation de taux
- CacheManager: Gestionnaire de cache
- SecurityUtils: Utilitaires de sécurité
"""

import time
import hashlib
import hmac
import secrets
import logging
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from functools import wraps, lru_cache
from dataclasses import dataclass
from collections import defaultdict
import asyncio

import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis.asyncio as aioredis


class Logger:
    """Système de logging avancé avec support multi-format."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Configuration du formatter
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def debug(self, message: str, **kwargs):
        """Log de debug."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log d'information."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log d'avertissement."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log d'erreur."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critique."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Log interne avec métadonnées."""
        if kwargs:
            # Ajouter les métadonnées au message
            extra_data = json.dumps(kwargs, default=str)
            message = f"{message} | Extra: {extra_data}"
        
        self.logger.log(level, message)
    
    def log_execution_time(self, func_name: str, execution_time: float, **kwargs):
        """Log le temps d'exécution d'une fonction."""
        self.info(
            f"Function {func_name} executed in {execution_time:.4f}s",
            function=func_name,
            execution_time=execution_time,
            **kwargs
        )
    
    def log_metric(self, metric_name: str, value: Union[int, float], tags: Optional[Dict] = None):
        """Log une métrique."""
        self.info(
            f"Metric {metric_name}: {value}",
            metric_name=metric_name,
            metric_value=value,
            metric_tags=tags or {}
        )


class Validator:
    """Validation de données avec règles personnalisées."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Valide un email."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_uuid(uuid_string: str) -> bool:
        """Valide un UUID."""
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(pattern, uuid_string.lower()))
    
    @staticmethod
    def validate_tenant_id(tenant_id: str) -> bool:
        """Valide un ID de tenant."""
        if not tenant_id or len(tenant_id) < 2:
            return False
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, tenant_id))
    
    @staticmethod
    def validate_metric_name(name: str) -> bool:
        """Valide un nom de métrique."""
        if not name or len(name) < 2:
            return False
        pattern = r'^[a-zA-Z][a-zA-Z0-9_.-]*$'
        return bool(re.match(pattern, name))
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, bool]:
        """Valide la force d'un mot de passe."""
        checks = {
            'length': len(password) >= 8,
            'uppercase': bool(re.search(r'[A-Z]', password)),
            'lowercase': bool(re.search(r'[a-z]', password)),
            'digit': bool(re.search(r'\d', password)),
            'special': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
        }
        checks['strong'] = all(checks.values())
        return checks
    
    @staticmethod
    def validate_json(data: str) -> bool:
        """Valide un JSON."""
        try:
            json.loads(data)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None, 
                      max_val: Optional[Union[int, float]] = None) -> bool:
        """Valide qu'une valeur est dans une plage."""
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 255) -> str:
        """Nettoie et limite une chaîne."""
        if not text:
            return ""
        
        # Supprime les caractères dangereux
        text = re.sub(r'[<>"\'\&]', '', text)
        
        # Limite la longueur
        return text[:max_length].strip()


class Formatter:
    """Formatage de données pour différents usages."""
    
    @staticmethod
    def format_bytes(bytes_value: int) -> str:
        """Formate des bytes en format lisible."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Formate une durée en format lisible."""
        if seconds < 1:
            return f"{seconds*1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def format_number(number: Union[int, float], precision: int = 2) -> str:
        """Formate un nombre avec séparateurs de milliers."""
        if isinstance(number, int):
            return f"{number:,}"
        return f"{number:,.{precision}f}"
    
    @staticmethod
    def format_percentage(value: float, total: float, precision: int = 1) -> str:
        """Formate un pourcentage."""
        if total == 0:
            return "0.0%"
        percentage = (value / total) * 100
        return f"{percentage:.{precision}f}%"
    
    @staticmethod
    def format_timestamp(timestamp: datetime, format_type: str = "iso") -> str:
        """Formate un timestamp."""
        formats = {
            "iso": timestamp.isoformat(),
            "readable": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H:%M:%S"),
            "compact": timestamp.strftime("%Y%m%d_%H%M%S")
        }
        return formats.get(format_type, timestamp.isoformat())
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Tronque un texte."""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix


class Timer:
    """Mesure de temps d'exécution."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0
    
    def start(self):
        """Démarre le chronomètre."""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self):
        """Arrête le chronomètre."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def __enter__(self):
        """Support du context manager."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Fin du context manager."""
        self.stop()
    
    @property
    def elapsed_formatted(self) -> str:
        """Temps écoulé formaté."""
        return Formatter.format_duration(self.elapsed)


@dataclass
class RateLimitInfo:
    """Information de rate limiting."""
    requests: int
    reset_time: datetime
    limit: int
    
    @property
    def remaining(self) -> int:
        """Requêtes restantes."""
        return max(0, self.limit - self.requests)
    
    @property
    def is_exceeded(self) -> bool:
        """Limite dépassée."""
        return self.requests >= self.limit


class RateLimiter:
    """Limitation de taux avec Redis backend."""
    
    def __init__(self, max_requests: int = 100, window: int = 3600, 
                 redis_client: Optional[aioredis.Redis] = None):
        self.max_requests = max_requests
        self.window = window  # en secondes
        self.redis_client = redis_client
        self.local_cache = defaultdict(list)  # Fallback local
    
    async def is_allowed(self, key: str) -> bool:
        """Vérifie si une requête est autorisée."""
        info = await self.get_rate_limit_info(key)
        return not info.is_exceeded
    
    async def get_rate_limit_info(self, key: str) -> RateLimitInfo:
        """Récupère les informations de rate limiting."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window)
        
        if self.redis_client:
            return await self._redis_rate_limit(key, now, window_start)
        else:
            return self._local_rate_limit(key, now, window_start)
    
    async def _redis_rate_limit(self, key: str, now: datetime, 
                              window_start: datetime) -> RateLimitInfo:
        """Rate limiting avec Redis."""
        redis_key = f"rate_limit:{key}"
        
        # Utiliser un pipeline pour l'atomicité
        pipe = self.redis_client.pipeline()
        
        # Supprimer les entrées expirées
        pipe.zremrangebyscore(redis_key, 0, window_start.timestamp())
        
        # Ajouter la requête actuelle
        pipe.zadd(redis_key, {str(now.timestamp()): now.timestamp()})
        
        # Compter les requêtes dans la fenêtre
        pipe.zcard(redis_key)
        
        # Définir l'expiration
        pipe.expire(redis_key, self.window)
        
        results = await pipe.execute()
        request_count = results[2]  # Résultat de zcard
        
        reset_time = now + timedelta(seconds=self.window)
        
        return RateLimitInfo(
            requests=request_count,
            reset_time=reset_time,
            limit=self.max_requests
        )
    
    def _local_rate_limit(self, key: str, now: datetime, 
                         window_start: datetime) -> RateLimitInfo:
        """Rate limiting local (fallback)."""
        # Nettoyer les anciennes entrées
        self.local_cache[key] = [
            timestamp for timestamp in self.local_cache[key]
            if timestamp > window_start
        ]
        
        # Ajouter la requête actuelle
        self.local_cache[key].append(now)
        
        reset_time = now + timedelta(seconds=self.window)
        
        return RateLimitInfo(
            requests=len(self.local_cache[key]),
            reset_time=reset_time,
            limit=self.max_requests
        )


class CacheManager:
    """Gestionnaire de cache avec TTL et invalidation."""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis_client = redis_client
        self.local_cache = {}
        self.local_ttl = {}
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur du cache."""
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value is not None:
                    return json.loads(value)
            except Exception:
                pass
        
        # Fallback local
        if key in self.local_cache:
            if key in self.local_ttl and datetime.utcnow() > self.local_ttl[key]:
                del self.local_cache[key]
                del self.local_ttl[key]
            else:
                return self.local_cache[key]
        
        return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Stocke une valeur dans le cache."""
        if self.redis_client:
            try:
                json_value = json.dumps(value, default=str)
                if ttl:
                    await self.redis_client.setex(key, ttl, json_value)
                else:
                    await self.redis_client.set(key, json_value)
                return
            except Exception:
                pass
        
        # Fallback local
        self.local_cache[key] = value
        if ttl:
            self.local_ttl[key] = datetime.utcnow() + timedelta(seconds=ttl)
    
    async def delete(self, key: str):
        """Supprime une valeur du cache."""
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception:
                pass
        
        # Local
        self.local_cache.pop(key, None)
        self.local_ttl.pop(key, None)
    
    async def clear(self, pattern: Optional[str] = None):
        """Vide le cache."""
        if self.redis_client and pattern:
            try:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception:
                pass
        elif self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception:
                pass
        
        # Local
        if pattern:
            import fnmatch
            keys_to_delete = [
                key for key in self.local_cache.keys()
                if fnmatch.fnmatch(key, pattern)
            ]
            for key in keys_to_delete:
                self.local_cache.pop(key, None)
                self.local_ttl.pop(key, None)
        else:
            self.local_cache.clear()
            self.local_ttl.clear()


class SecurityUtils:
    """Utilitaires de sécurité."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hache un mot de passe avec bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Vérifie un mot de passe."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Génère un token sécurisé."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_api_key() -> str:
        """Génère une clé API."""
        return f"spa_{SecurityUtils.generate_token(32)}"
    
    @staticmethod
    def create_hmac_signature(data: str, secret: str) -> str:
        """Crée une signature HMAC."""
        return hmac.new(
            secret.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    @staticmethod
    def verify_hmac_signature(data: str, signature: str, secret: str) -> bool:
        """Vérifie une signature HMAC."""
        expected = SecurityUtils.create_hmac_signature(data, secret)
        return hmac.compare_digest(signature, expected)
    
    @staticmethod
    def encrypt_data(data: str, key: str) -> str:
        """Chiffre des données avec Fernet."""
        # Dériver une clé à partir de la clé fournie
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'salt_analytics',
            iterations=100000,
        )
        derived_key = kdf.derive(key.encode())
        fernet = Fernet(Fernet.generate_key())
        return fernet.encrypt(data.encode()).decode()
    
    @staticmethod
    def decrypt_data(encrypted_data: str, key: str) -> str:
        """Déchiffre des données."""
        # À implémenter selon les besoins
        return encrypted_data
    
    @staticmethod
    def mask_sensitive_data(data: str, show_chars: int = 4) -> str:
        """Masque des données sensibles."""
        if len(data) <= show_chars * 2:
            return "*" * len(data)
        return data[:show_chars] + "*" * (len(data) - show_chars * 2) + data[-show_chars:]


def retry_async(max_attempts: int = 3, delay: float = 1.0, 
                exponential_backoff: bool = True):
    """Décorateur pour retry automatique des fonctions async."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                        await asyncio.sleep(wait_time)
                    
            raise last_exception
        return wrapper
    return decorator


def measure_time(func: Callable):
    """Décorateur pour mesurer le temps d'exécution."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        with Timer() as timer:
            result = await func(*args, **kwargs)
        
        logger = Logger(func.__module__)
        logger.log_execution_time(func.__name__, timer.elapsed)
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        with Timer() as timer:
            result = func(*args, **kwargs)
        
        logger = Logger(func.__module__)
        logger.log_execution_time(func.__name__, timer.elapsed)
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


@lru_cache(maxsize=128)
def get_cached_result(cache_key: str, func: Callable, *args, **kwargs):
    """Cache LRU pour les résultats de fonction."""
    return func(*args, **kwargs)


def create_correlation_id() -> str:
    """Crée un ID de corrélation unique."""
    return f"corr_{int(time.time())}_{secrets.token_hex(8)}"


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Charge du JSON de manière sécurisée."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Fusionne deux dictionnaires en profondeur."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Divise une liste en chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Aplatit un dictionnaire imbriqué."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
