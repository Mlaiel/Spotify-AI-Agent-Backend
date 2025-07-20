"""
Utilitaires avancés pour la configuration Alertmanager Receivers

Ce module fournit des fonctions utilitaires robustes pour la manipulation
des configurations, le chiffrement, la validation et les transformations.

Author: Spotify AI Agent Team
Maintainer: Fahed Mlaiel - DBA & Data Engineer
"""

import logging
import re
import json
import yaml
import base64
import hashlib
import secrets
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import asdict, is_dataclass
from urllib.parse import urlparse, parse_qs
import ipaddress
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# ============================================================================
# UTILITAIRES DE CONFIGURATION
# ============================================================================

class ConfigUtils:
    """Utilitaires pour la manipulation des configurations"""
    
    @staticmethod
    def load_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Charge un fichier de configuration YAML ou JSON"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as file:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(file)
                elif path.suffix.lower() == '.json':
                    return json.load(file)
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")
                    
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
            raise
    
    @staticmethod
    def save_config_file(config: Dict[str, Any], file_path: Union[str, Path], format: str = 'yaml'):
        """Sauvegarde une configuration dans un fichier"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as file:
                if format.lower() in ['yaml', 'yml']:
                    yaml.dump(config, file, default_flow_style=False, indent=2, sort_keys=False)
                elif format.lower() == 'json':
                    json.dump(config, file, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                    
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config file {file_path}: {e}")
            raise
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fusionne deux configurations de manière récursive"""
        def _merge_recursive(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = _merge_recursive(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        return _merge_recursive(base_config, override_config)
    
    @staticmethod
    def expand_environment_variables(config: Dict[str, Any]) -> Dict[str, Any]:
        """Remplace les variables d'environnement dans la configuration"""
        import os
        
        def _expand_value(value: Any) -> Any:
            if isinstance(value, str):
                # Remplace ${VAR} et ${VAR:default}
                pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
                
                def replace_var(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ""
                    return os.getenv(var_name, default_value)
                
                return re.sub(pattern, replace_var, value)
            elif isinstance(value, dict):
                return {k: _expand_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_expand_value(item) for item in value]
            else:
                return value
        
        return _expand_value(config)
    
    @staticmethod
    def validate_config_structure(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """Valide la structure d'une configuration"""
        missing_keys = []
        
        def _check_nested_key(obj: Dict, key_path: str):
            keys = key_path.split('.')
            current = obj
            
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return False
                current = current[key]
            
            return True
        
        for key in required_keys:
            if not _check_nested_key(config, key):
                missing_keys.append(key)
        
        return missing_keys
    
    @staticmethod
    def sanitize_tenant_name(name: str) -> str:
        """Nettoie et normalise un nom de tenant"""
        # Conversion en minuscules
        sanitized = name.lower()
        
        # Remplacement des caractères non autorisés par des tirets
        sanitized = re.sub(r'[^a-z0-9-]', '-', sanitized)
        
        # Suppression des tirets multiples
        sanitized = re.sub(r'-+', '-', sanitized)
        
        # Suppression des tirets en début/fin
        sanitized = sanitized.strip('-')
        
        # Limitation de la longueur
        if len(sanitized) > 50:
            sanitized = sanitized[:50].rstrip('-')
        
        return sanitized
    
    @staticmethod
    def generate_config_id(config: Dict[str, Any]) -> str:
        """Génère un ID unique pour une configuration"""
        config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def extract_secrets_from_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Extrait les secrets d'une configuration"""
        secrets_map = {}
        clean_config = {}
        
        secret_patterns = [
            r'.*key.*',
            r'.*password.*',
            r'.*token.*',
            r'.*secret.*',
            r'.*credential.*'
        ]
        
        def _extract_recursive(obj: Any, path: str = "") -> Any:
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Vérification si c'est un secret
                    is_secret = any(re.match(pattern, key.lower()) for pattern in secret_patterns)
                    
                    if is_secret and isinstance(value, str):
                        secret_id = f"SECRET_{len(secrets_map) + 1}"
                        secrets_map[secret_id] = value
                        result[key] = f"${{{secret_id}}}"
                    else:
                        result[key] = _extract_recursive(value, current_path)
                
                return result
            elif isinstance(obj, list):
                return [_extract_recursive(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            else:
                return obj
        
        clean_config = _extract_recursive(config)
        return clean_config, secrets_map

# ============================================================================
# UTILITAIRES DE CHIFFREMENT
# ============================================================================

class EncryptionUtils:
    """Utilitaires pour le chiffrement et la sécurité"""
    
    @staticmethod
    def generate_key() -> bytes:
        """Génère une clé de chiffrement sécurisée"""
        return Fernet.generate_key()
    
    @staticmethod
    def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Dérive une clé à partir d'un mot de passe"""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    @staticmethod
    def encrypt_data(data: str, key: bytes) -> str:
        """Chiffre des données avec une clé"""
        try:
            fernet = Fernet(key)
            encrypted_data = fernet.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    @staticmethod
    def decrypt_data(encrypted_data: str, key: bytes) -> str:
        """Déchiffre des données avec une clé"""
        try:
            fernet = Fernet(key)
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    @staticmethod
    def hash_data(data: str, algorithm: str = 'sha256') -> str:
        """Calcule le hash d'une donnée"""
        if algorithm == 'sha256':
            return hashlib.sha256(data.encode()).hexdigest()
        elif algorithm == 'sha512':
            return hashlib.sha512(data.encode()).hexdigest()
        elif algorithm == 'md5':
            return hashlib.md5(data.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    @staticmethod
    def verify_hash(data: str, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """Vérifie le hash d'une donnée"""
        actual_hash = EncryptionUtils.hash_data(data, algorithm)
        return secrets.compare_digest(actual_hash, expected_hash)
    
    @staticmethod
    def generate_random_string(length: int = 32, include_symbols: bool = False) -> str:
        """Génère une chaîne aléatoire sécurisée"""
        import string
        
        alphabet = string.ascii_letters + string.digits
        if include_symbols:
            alphabet += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def generate_api_key(prefix: str = "spa", length: int = 32) -> str:
        """Génère une clé API avec préfixe"""
        random_part = EncryptionUtils.generate_random_string(length)
        return f"{prefix}_{random_part}"
    
    @staticmethod
    def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
        """Masque les données sensibles en gardant quelques caractères visibles"""
        if len(data) <= visible_chars * 2:
            return "*" * len(data)
        
        return data[:visible_chars] + "*" * (len(data) - visible_chars * 2) + data[-visible_chars:]

# ============================================================================
# UTILITAIRES DE VALIDATION
# ============================================================================

class ValidationUtils:
    """Utilitaires pour la validation de données"""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Valide une adresse email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def is_valid_url(url: str, schemes: Optional[List[str]] = None) -> bool:
        """Valide une URL"""
        if schemes is None:
            schemes = ['http', 'https']
        
        try:
            result = urlparse(url)
            return all([
                result.scheme in schemes,
                result.netloc,
                len(result.netloc) > 0
            ])
        except:
            return False
    
    @staticmethod
    def is_valid_ip_address(ip: str) -> bool:
        """Valide une adresse IP (v4 ou v6)"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_valid_cidr(cidr: str) -> bool:
        """Valide une notation CIDR"""
        try:
            ipaddress.ip_network(cidr, strict=False)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_valid_duration(duration: str) -> bool:
        """Valide un format de durée (ex: 5m, 1h, 30s)"""
        pattern = r'^\d+[smhd]$'
        return bool(re.match(pattern, duration))
    
    @staticmethod
    def parse_duration_to_seconds(duration: str) -> int:
        """Convertit une durée en secondes"""
        if not ValidationUtils.is_valid_duration(duration):
            raise ValueError(f"Invalid duration format: {duration}")
        
        value = int(duration[:-1])
        unit = duration[-1]
        
        multipliers = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400
        }
        
        return value * multipliers[unit]
    
    @staticmethod
    def is_valid_json(json_str: str) -> bool:
        """Valide une chaîne JSON"""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False
    
    @staticmethod
    def is_valid_yaml(yaml_str: str) -> bool:
        """Valide une chaîne YAML"""
        try:
            yaml.safe_load(yaml_str)
            return True
        except yaml.YAMLError:
            return False
    
    @staticmethod
    def validate_port_number(port: Union[str, int]) -> bool:
        """Valide un numéro de port"""
        try:
            port_num = int(port)
            return 1 <= port_num <= 65535
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_webhook_url(url: str, required_host: Optional[str] = None) -> bool:
        """Valide une URL de webhook avec des critères spécifiques"""
        if not ValidationUtils.is_valid_url(url, ['https']):
            return False
        
        parsed = urlparse(url)
        
        # Vérification de l'hôte si spécifié
        if required_host and parsed.hostname != required_host:
            return False
        
        # Vérification que l'URL contient un chemin
        if not parsed.path or parsed.path == '/':
            return False
        
        return True

# ============================================================================
# UTILITAIRES DE TRANSFORMATION
# ============================================================================

class TransformationUtils:
    """Utilitaires pour la transformation de données"""
    
    @staticmethod
    def flatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
        """Aplatit un dictionnaire imbriqué"""
        def _flatten_recursive(obj: Any, parent_key: str = '') -> Dict[str, Any]:
            items = []
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{parent_key}{separator}{key}" if parent_key else key
                    
                    if isinstance(value, (dict, list)):
                        items.extend(_flatten_recursive(value, new_key).items())
                    else:
                        items.append((new_key, value))
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    new_key = f"{parent_key}{separator}{i}" if parent_key else str(i)
                    
                    if isinstance(value, (dict, list)):
                        items.extend(_flatten_recursive(value, new_key).items())
                    else:
                        items.append((new_key, value))
            else:
                items.append((parent_key, obj))
            
            return dict(items)
        
        return _flatten_recursive(data)
    
    @staticmethod
    def unflatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
        """Reconstitue un dictionnaire à partir d'un dictionnaire aplati"""
        result = {}
        
        for key, value in data.items():
            keys = key.split(separator)
            current = result
            
            for i, k in enumerate(keys[:-1]):
                if k.isdigit():
                    # C'est un index de liste
                    k = int(k)
                    if not isinstance(current, list):
                        current = []
                    
                    # Étendre la liste si nécessaire
                    while len(current) <= k:
                        current.append({})
                    
                    if isinstance(current[k], dict) and keys[i + 1].isdigit():
                        current[k] = []
                    elif not isinstance(current[k], (dict, list)):
                        current[k] = {}
                    
                    current = current[k]
                else:
                    # C'est une clé de dictionnaire
                    if k not in current:
                        # Déterminer si le prochain élément est un index de liste
                        next_key = keys[i + 1] if i + 1 < len(keys) - 1 else keys[-1]
                        current[k] = [] if next_key.isdigit() else {}
                    
                    current = current[k]
            
            # Définir la valeur finale
            final_key = keys[-1]
            if final_key.isdigit():
                final_key = int(final_key)
                if not isinstance(current, list):
                    current = []
                
                while len(current) <= final_key:
                    current.append(None)
                
                current[final_key] = value
            else:
                current[final_key] = value
        
        return result
    
    @staticmethod
    def convert_dataclass_to_dict(obj: Any) -> Any:
        """Convertit un dataclass en dictionnaire récursivement"""
        if is_dataclass(obj):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: TransformationUtils.convert_dataclass_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [TransformationUtils.convert_dataclass_to_dict(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def deep_merge_lists(list1: List[Any], list2: List[Any]) -> List[Any]:
        """Fusionne deux listes de manière intelligente"""
        result = list1.copy()
        
        for item in list2:
            if item not in result:
                result.append(item)
        
        return result
    
    @staticmethod
    def filter_dict_by_keys(data: Dict[str, Any], allowed_keys: List[str]) -> Dict[str, Any]:
        """Filtre un dictionnaire en gardant seulement les clés autorisées"""
        return {k: v for k, v in data.items() if k in allowed_keys}
    
    @staticmethod
    def remove_empty_values(data: Dict[str, Any]) -> Dict[str, Any]:
        """Supprime les valeurs vides d'un dictionnaire"""
        def _is_empty(value: Any) -> bool:
            return (
                value is None or
                value == "" or
                value == [] or
                value == {} or
                (isinstance(value, dict) and all(_is_empty(v) for v in value.values()))
            )
        
        return {k: v for k, v in data.items() if not _is_empty(v)}

# ============================================================================
# UTILITAIRES ASYNCRONES
# ============================================================================

class AsyncUtils:
    """Utilitaires pour la programmation asynchrone"""
    
    @staticmethod
    async def run_with_timeout(coro: Callable, timeout_seconds: float, *args, **kwargs) -> Any:
        """Exécute une coroutine avec timeout"""
        try:
            return await asyncio.wait_for(coro(*args, **kwargs), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout_seconds} seconds")
            raise
    
    @staticmethod
    async def batch_process(
        items: List[Any], 
        processor: Callable,
        batch_size: int = 10,
        max_concurrent: int = 5
    ) -> List[Any]:
        """Traite une liste d'éléments par lots avec concurrence limitée"""
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch: List[Any]) -> List[Any]:
            async with semaphore:
                tasks = [processor(item) for item in batch]
                return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Découpage en lots
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    @staticmethod
    async def retry_async(
        func: Callable,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: Tuple = (Exception,)
    ) -> Any:
        """Retry une fonction asynchrone avec backoff exponentiel"""
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except exceptions as e:
                if attempt == max_retries:
                    raise e
                
                wait_time = delay * (backoff_factor ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

# ============================================================================
# UTILITAIRES DE PERFORMANCE
# ============================================================================

class PerformanceUtils:
    """Utilitaires pour l'optimisation des performances"""
    
    @staticmethod
    def timeit_sync(func: Callable) -> Callable:
        """Décorateur pour mesurer le temps d'exécution synchrone"""
        import time
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return wrapper
    
    @staticmethod
    def timeit_async(func: Callable) -> Callable:
        """Décorateur pour mesurer le temps d'exécution asynchrone"""
        import time
        import functools
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return wrapper
    
    @staticmethod
    def memoize(maxsize: int = 128):
        """Décorateur de mise en cache avec taille limitée"""
        from functools import lru_cache
        return lru_cache(maxsize=maxsize)
    
    @staticmethod
    def rate_limit(calls_per_second: float):
        """Décorateur pour limiter le taux d'appel"""
        import time
        import functools
        
        min_interval = 1.0 / calls_per_second
        last_called = [0.0]
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                elapsed = time.time() - last_called[0]
                left_to_wait = min_interval - elapsed
                
                if left_to_wait > 0:
                    time.sleep(left_to_wait)
                
                ret = func(*args, **kwargs)
                last_called[0] = time.time()
                return ret
            
            return wrapper
        
        return decorator

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ConfigUtils",
    "EncryptionUtils", 
    "ValidationUtils",
    "TransformationUtils",
    "AsyncUtils",
    "PerformanceUtils"
]
