"""
Script Utilities Module - Advanced Utility Functions

This module provides comprehensive utility functions for the scripts ecosystem,
including validation, formatting, encryption, compression, and system operations.

Version: 3.0.0
Developed by Spotify AI Agent Team
"""

import asyncio
import logging
import json
import yaml
import hashlib
import hmac
import secrets
import base64
import gzip
import lzma
import brotli
import os
import sys
import time
import re
import ipaddress
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import aiofiles
import aiohttp
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis.asyncio as redis
import psutil
import socket
import subprocess

logger = logging.getLogger(__name__)

class DataFormat(Enum):
    """Formats de données supportés"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    XML = "xml"
    CSV = "csv"
    BINARY = "binary"

class CompressionType(Enum):
    """Types de compression supportés"""
    GZIP = "gzip"
    LZMA = "lzma"
    BROTLI = "brotli"
    LZ4 = "lz4"
    ZSTD = "zstd"

class EncryptionAlgorithm(Enum):
    """Algorithmes de chiffrement supportés"""
    FERNET = "fernet"
    AES_256 = "aes_256"
    CHACHA20 = "chacha20"

# ============================================================================
# Validation Utilities
# ============================================================================

class ValidationError(Exception):
    """Exception pour les erreurs de validation"""
    pass

def validate_ip_address(ip: str) -> bool:
    """Valide une adresse IP (IPv4 ou IPv6)"""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

def validate_port(port: Union[str, int]) -> bool:
    """Valide un numéro de port"""
    try:
        port_int = int(port)
        return 1 <= port_int <= 65535
    except (ValueError, TypeError):
        return False

def validate_url(url: str) -> bool:
    """Valide une URL"""
    url_pattern = re.compile(
        r'^https?://'  # http:// ou https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domaine
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # port optionnel
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def validate_email(email: str) -> bool:
    """Valide une adresse email"""
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    return email_pattern.match(email) is not None

def validate_json(data: str) -> bool:
    """Valide une chaîne JSON"""
    try:
        json.loads(data)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def validate_yaml(data: str) -> bool:
    """Valide une chaîne YAML"""
    try:
        yaml.safe_load(data)
        return True
    except yaml.YAMLError:
        return False

def validate_kubernetes_name(name: str) -> bool:
    """Valide un nom Kubernetes (DNS-1123 subdomain)"""
    if not name or len(name) > 253:
        return False
    
    pattern = re.compile(r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')
    return pattern.match(name) is not None

def validate_docker_image_tag(tag: str) -> bool:
    """Valide un tag d'image Docker"""
    if not tag or len(tag) > 128:
        return False
    
    # Pattern simplifié pour les tags Docker
    pattern = re.compile(r'^[a-zA-Z0-9._-]+$')
    return pattern.match(tag) is not None

# ============================================================================
# Data Format Utilities
# ============================================================================

async def serialize_data(data: Any, format_type: DataFormat) -> str:
    """Sérialise des données dans le format spécifié"""
    
    if format_type == DataFormat.JSON:
        return json.dumps(data, indent=2, default=str)
    elif format_type == DataFormat.YAML:
        return yaml.dump(data, default_flow_style=False)
    elif format_type == DataFormat.TOML:
        import toml
        return toml.dumps(data)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

async def deserialize_data(data: str, format_type: DataFormat) -> Any:
    """Désérialise des données depuis le format spécifié"""
    
    if format_type == DataFormat.JSON:
        return json.loads(data)
    elif format_type == DataFormat.YAML:
        return yaml.safe_load(data)
    elif format_type == DataFormat.TOML:
        import toml
        return toml.loads(data)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def convert_data_format(data: str, from_format: DataFormat, to_format: DataFormat) -> str:
    """Convertit des données d'un format à un autre"""
    
    # Désérialisation
    parsed_data = asyncio.run(deserialize_data(data, from_format))
    
    # Sérialisation dans le nouveau format
    return asyncio.run(serialize_data(parsed_data, to_format))

# ============================================================================
# Compression Utilities
# ============================================================================

def compress_data(data: bytes, compression_type: CompressionType) -> bytes:
    """Compresse des données avec l'algorithme spécifié"""
    
    if compression_type == CompressionType.GZIP:
        return gzip.compress(data)
    elif compression_type == CompressionType.LZMA:
        return lzma.compress(data)
    elif compression_type == CompressionType.BROTLI:
        return brotli.compress(data)
    elif compression_type == CompressionType.LZ4:
        import lz4.frame
        return lz4.frame.compress(data)
    elif compression_type == CompressionType.ZSTD:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor()
        return cctx.compress(data)
    else:
        raise ValueError(f"Unsupported compression type: {compression_type}")

def decompress_data(data: bytes, compression_type: CompressionType) -> bytes:
    """Décompresse des données avec l'algorithme spécifié"""
    
    if compression_type == CompressionType.GZIP:
        return gzip.decompress(data)
    elif compression_type == CompressionType.LZMA:
        return lzma.decompress(data)
    elif compression_type == CompressionType.BROTLI:
        return brotli.decompress(data)
    elif compression_type == CompressionType.LZ4:
        import lz4.frame
        return lz4.frame.decompress(data)
    elif compression_type == CompressionType.ZSTD:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    else:
        raise ValueError(f"Unsupported compression type: {compression_type}")

def get_optimal_compression(data: bytes) -> CompressionType:
    """Détermine l'algorithme de compression optimal pour les données"""
    
    data_size = len(data)
    
    # Test de différents algorithmes
    compression_results = {}
    
    for comp_type in CompressionType:
        try:
            compressed = compress_data(data, comp_type)
            compression_ratio = len(compressed) / data_size
            compression_results[comp_type] = compression_ratio
        except Exception:
            continue
    
    # Retourner l'algorithme avec le meilleur ratio
    if compression_results:
        return min(compression_results, key=compression_results.get)
    else:
        return CompressionType.GZIP  # Fallback

# ============================================================================
# Encryption Utilities
# ============================================================================

def generate_encryption_key(algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET) -> bytes:
    """Génère une clé de chiffrement"""
    
    if algorithm == EncryptionAlgorithm.FERNET:
        return Fernet.generate_key()
    elif algorithm == EncryptionAlgorithm.AES_256:
        return secrets.token_bytes(32)  # 256 bits
    elif algorithm == EncryptionAlgorithm.CHACHA20:
        return secrets.token_bytes(32)  # 256 bits
    else:
        raise ValueError(f"Unsupported encryption algorithm: {algorithm}")

def encrypt_data(data: bytes, key: bytes, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET) -> bytes:
    """Chiffre des données"""
    
    if algorithm == EncryptionAlgorithm.FERNET:
        f = Fernet(key)
        return f.encrypt(data)
    elif algorithm == EncryptionAlgorithm.AES_256:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        # Padding PKCS7
        pad_len = 16 - (len(data) % 16)
        padded_data = data + bytes([pad_len]) * pad_len
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        return iv + encrypted
    else:
        raise ValueError(f"Unsupported encryption algorithm: {algorithm}")

def decrypt_data(encrypted_data: bytes, key: bytes, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET) -> bytes:
    """Déchiffre des données"""
    
    if algorithm == EncryptionAlgorithm.FERNET:
        f = Fernet(key)
        return f.decrypt(encrypted_data)
    elif algorithm == EncryptionAlgorithm.AES_256:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        iv = encrypted_data[:16]
        encrypted = encrypted_data[16:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted) + decryptor.finalize()
        # Retirer le padding
        pad_len = padded_data[-1]
        return padded_data[:-pad_len]
    else:
        raise ValueError(f"Unsupported encryption algorithm: {algorithm}")

# ============================================================================
# Hash and Checksum Utilities
# ============================================================================

def calculate_hash(data: bytes, algorithm: str = "sha256") -> str:
    """Calcule le hash des données"""
    
    if algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(data).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

async def calculate_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Calcule le hash d'un fichier de manière asynchrone"""
    
    hash_obj = getattr(hashlib, algorithm)()
    
    async with aiofiles.open(file_path, 'rb') as f:
        while chunk := await f.read(8192):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()

def verify_hmac(data: bytes, key: bytes, signature: str, algorithm: str = "sha256") -> bool:
    """Vérifie une signature HMAC"""
    
    expected_signature = hmac.new(key, data, getattr(hashlib, algorithm)).hexdigest()
    return hmac.compare_digest(expected_signature, signature)

# ============================================================================
# System Utilities
# ============================================================================

def get_system_info() -> Dict[str, Any]:
    """Récupère les informations système"""
    
    return {
        "platform": sys.platform,
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_usage": dict(psutil.disk_usage('/')),
        "boot_time": psutil.boot_time(),
        "hostname": socket.gethostname(),
        "ip_addresses": get_local_ip_addresses()
    }

def get_local_ip_addresses() -> List[str]:
    """Récupère les adresses IP locales"""
    
    addresses = []
    
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:  # IPv4
                addresses.append(addr.address)
    
    return addresses

async def execute_command(command: str, timeout: int = 300) -> Dict[str, Any]:
    """Exécute une commande système de manière asynchrone"""
    
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), 
            timeout=timeout
        )
        
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode('utf-8'),
            "stderr": stderr.decode('utf-8'),
            "success": process.returncode == 0
        }
        
    except asyncio.TimeoutError:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "Command timed out",
            "success": False,
            "timeout": True
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "error": str(e)
        }

def check_port_open(host: str, port: int, timeout: float = 3.0) -> bool:
    """Vérifie si un port est ouvert"""
    
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.error, socket.timeout):
        return False

async def wait_for_service(host: str, port: int, timeout: int = 300, interval: float = 1.0) -> bool:
    """Attend qu'un service soit disponible"""
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if check_port_open(host, port):
            return True
        await asyncio.sleep(interval)
    
    return False

# ============================================================================
# String and Text Utilities
# ============================================================================

def sanitize_string(text: str, allowed_chars: str = None) -> str:
    """Nettoie une chaîne de caractères"""
    
    if allowed_chars is None:
        # Caractères alphanumériques + quelques caractères spéciaux
        allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    
    return ''.join(char for char in text if char in allowed_chars)

def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Tronque une chaîne à la longueur spécifiée"""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def format_bytes(bytes_value: int) -> str:
    """Formate une taille en bytes en format lisible"""
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    
    return f"{bytes_value:.1f} PB"

def format_duration(seconds: float) -> str:
    """Formate une durée en secondes en format lisible"""
    
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

# ============================================================================
# Configuration Utilities
# ============================================================================

class ConfigManager:
    """Gestionnaire de configuration centralisé"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config.yaml")
        self._config = {}
        self._watchers = []
    
    async def load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier"""
        
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        
        try:
            async with aiofiles.open(self.config_path, 'r') as f:
                content = await f.read()
            
            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                self._config = yaml.safe_load(content)
            elif self.config_path.suffix.lower() == '.json':
                self._config = json.loads(content)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de configuration"""
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Définit une valeur de configuration"""
        
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    async def save_config(self):
        """Sauvegarde la configuration dans le fichier"""
        
        try:
            content = yaml.dump(self._config, default_flow_style=False)
            
            async with aiofiles.open(self.config_path, 'w') as f:
                await f.write(content)
                
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise

# ============================================================================
# Rate Limiting Utilities
# ============================================================================

class RateLimiter:
    """Limiteur de débit avec algorithme de seau à tokens"""
    
    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = max_tokens
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquiert des tokens (retourne True si possible)"""
        
        async with self.lock:
            now = time.time()
            
            # Remplissage du seau
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Vérification de la disponibilité
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None):
        """Attend que les tokens soient disponibles"""
        
        start_time = time.time()
        
        while True:
            if await self.acquire(tokens):
                return
            
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError("Rate limit timeout")
            
            await asyncio.sleep(0.1)

# ============================================================================
# Retry Utilities
# ============================================================================

def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,)
):
    """Décorateur pour retry automatique de fonctions async"""
    
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator

# ============================================================================
# Cache Utilities
# ============================================================================

class AsyncCache:
    """Cache asynchrone avec TTL"""
    
    def __init__(self, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self._cache = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    return value
                else:
                    del self._cache[key]
            
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Stocke une valeur dans le cache"""
        
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        async with self._lock:
            self._cache[key] = (value, expiry)
    
    async def delete(self, key: str):
        """Supprime une valeur du cache"""
        
        async with self._lock:
            self._cache.pop(key, None)
    
    async def clear(self):
        """Vide le cache"""
        
        async with self._lock:
            self._cache.clear()
    
    async def cleanup_expired(self):
        """Nettoie les entrées expirées"""
        
        current_time = time.time()
        
        async with self._lock:
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if current_time >= expiry
            ]
            
            for key in expired_keys:
                del self._cache[key]

# Instances globales
config_manager = ConfigManager()
rate_limiter = RateLimiter(max_tokens=100, refill_rate=10.0)
cache = AsyncCache()

# Exports principaux
__all__ = [
    # Validation
    "validate_ip_address", "validate_port", "validate_url", "validate_email",
    "validate_json", "validate_yaml", "validate_kubernetes_name", "validate_docker_image_tag",
    
    # Data formats
    "serialize_data", "deserialize_data", "convert_data_format",
    
    # Compression
    "compress_data", "decompress_data", "get_optimal_compression",
    
    # Encryption
    "generate_encryption_key", "encrypt_data", "decrypt_data",
    
    # Hash
    "calculate_hash", "calculate_file_hash", "verify_hmac",
    
    # System
    "get_system_info", "execute_command", "check_port_open", "wait_for_service",
    
    # Text
    "sanitize_string", "truncate_string", "format_bytes", "format_duration",
    
    # Classes
    "ConfigManager", "RateLimiter", "AsyncCache",
    
    # Decorators
    "retry_async",
    
    # Enums
    "DataFormat", "CompressionType", "EncryptionAlgorithm",
    
    # Exceptions
    "ValidationError",
    
    # Globals
    "config_manager", "rate_limiter", "cache"
]
