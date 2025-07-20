"""
Security Utilities Module
========================

Ce module fournit des utilitaires de sécurité pour le système multi-tenant
du Spotify AI Agent.

Auteur: Fahed Mlaiel
"""

import asyncio
import hashlib
import hmac
import secrets
import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
import re
import uuid
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
import bcrypt
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class HashAlgorithm(Enum):
    """Algorithmes de hachage supportés"""
    SHA256 = "sha256"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


class EncryptionAlgorithm(Enum):
    """Algorithmes de chiffrement supportés"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"


@dataclass
class SecurityToken:
    """Token de sécurité"""
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_value: str = ""
    token_type: str = "bearer"
    
    # Métadonnées
    tenant_id: str = ""
    user_id: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    
    # Expiration
    issued_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Sécurité
    is_revoked: bool = False
    revoked_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    
    # Métadonnées d'audit
    created_by: str = ""
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class EncryptionKey:
    """Clé de chiffrement"""
    key_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_data: bytes = b""
    
    # Métadonnées
    tenant_id: str = ""
    purpose: str = "general"
    
    # Rotation
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    
    # Hiérarchie des clés
    parent_key_id: Optional[str] = None
    derivation_info: Optional[str] = None


class PasswordHasher:
    """
    Gestionnaire de hachage de mots de passe sécurisé
    """
    
    def __init__(self, rounds: int = 12):
        self.rounds = rounds
        
    def hash_password(self, password: str) -> str:
        """Hache un mot de passe avec bcrypt"""
        salt = bcrypt.gensalt(rounds=self.rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Vérifie un mot de passe contre son hash"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def is_strong_password(self, password: str) -> Tuple[bool, List[str]]:
        """Vérifie la force d'un mot de passe"""
        issues = []
        
        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        
        if len(password) > 128:
            issues.append("Password must be less than 128 characters long")
        
        if not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'\d', password):
            issues.append("Password must contain at least one digit")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one special character")
        
        # Vérification des patterns communs
        common_patterns = [
            r'123456',
            r'password',
            r'qwerty',
            r'abc123',
            r'admin'
        ]
        
        for pattern in common_patterns:
            if re.search(pattern, password.lower()):
                issues.append("Password contains common patterns")
                break
        
        return len(issues) == 0, issues


class SecretGenerator:
    """
    Générateur de secrets cryptographiquement sécurisés
    """
    
    @staticmethod
    def generate_random_string(length: int = 32, alphabet: str = None) -> str:
        """Génère une chaîne aléatoire sécurisée"""
        if alphabet is None:
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def generate_api_key(prefix: str = "ak", length: int = 32) -> str:
        """Génère une clé API"""
        random_part = SecretGenerator.generate_random_string(length)
        return f"{prefix}_{random_part}"
    
    @staticmethod
    def generate_token(length: int = 64) -> str:
        """Génère un token aléatoire"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_salt(length: int = 32) -> bytes:
        """Génère un salt cryptographique"""
        return secrets.token_bytes(length)
    
    @staticmethod
    def generate_encryption_key(algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM) -> bytes:
        """Génère une clé de chiffrement"""
        if algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC]:
            return secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.FERNET:
            return Fernet.generate_key()
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")


class DataEncryption:
    """
    Gestionnaire de chiffrement de données
    """
    
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self.fernet = Fernet(base64.urlsafe_b64encode(master_key[:32]))
    
    def encrypt_with_aes_gcm(self, plaintext: bytes, key: bytes, associated_data: bytes = b"") -> Dict[str, bytes]:
        """Chiffre avec AES-GCM"""
        # Génération d'un IV aléatoire
        iv = secrets.token_bytes(12)  # 96 bits pour GCM
        
        # Configuration du chiffrement
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        # Chiffrement
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return {
            "ciphertext": ciphertext,
            "iv": iv,
            "tag": encryptor.tag,
            "associated_data": associated_data
        }
    
    def decrypt_with_aes_gcm(self, encrypted_data: Dict[str, bytes], key: bytes) -> bytes:
        """Déchiffre avec AES-GCM"""
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(encrypted_data["iv"], encrypted_data["tag"])
        )
        decryptor = cipher.decryptor()
        
        if encrypted_data["associated_data"]:
            decryptor.authenticate_additional_data(encrypted_data["associated_data"])
        
        return decryptor.update(encrypted_data["ciphertext"]) + decryptor.finalize()
    
    def encrypt_json(self, data: Dict[str, Any], tenant_id: str) -> str:
        """Chiffre des données JSON pour un tenant"""
        json_str = json.dumps(data, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        
        # Utilisation des données associées pour le tenant
        associated_data = f"tenant:{tenant_id}".encode('utf-8')
        
        encrypted = self.encrypt_with_aes_gcm(
            json_bytes,
            self.master_key[:32],
            associated_data
        )
        
        # Encodage base64 du résultat
        result = {
            "ciphertext": base64.b64encode(encrypted["ciphertext"]).decode('ascii'),
            "iv": base64.b64encode(encrypted["iv"]).decode('ascii'),
            "tag": base64.b64encode(encrypted["tag"]).decode('ascii'),
            "associated_data": base64.b64encode(encrypted["associated_data"]).decode('ascii')
        }
        
        return base64.b64encode(json.dumps(result).encode('utf-8')).decode('ascii')
    
    def decrypt_json(self, encrypted_data: str, tenant_id: str) -> Dict[str, Any]:
        """Déchiffre des données JSON pour un tenant"""
        # Décodage base64
        decoded = json.loads(base64.b64decode(encrypted_data.encode('ascii')))
        
        encrypted_dict = {
            "ciphertext": base64.b64decode(decoded["ciphertext"]),
            "iv": base64.b64decode(decoded["iv"]),
            "tag": base64.b64decode(decoded["tag"]),
            "associated_data": base64.b64decode(decoded["associated_data"])
        }
        
        # Vérification du tenant dans les données associées
        expected_associated_data = f"tenant:{tenant_id}".encode('utf-8')
        if encrypted_dict["associated_data"] != expected_associated_data:
            raise ValueError("Invalid tenant for encrypted data")
        
        decrypted_bytes = self.decrypt_with_aes_gcm(encrypted_dict, self.master_key[:32])
        return json.loads(decrypted_bytes.decode('utf-8'))
    
    def encrypt_field(self, value: str, field_name: str, tenant_id: str) -> str:
        """Chiffre un champ spécifique"""
        associated_data = f"field:{field_name}:tenant:{tenant_id}".encode('utf-8')
        
        encrypted = self.encrypt_with_aes_gcm(
            value.encode('utf-8'),
            self.master_key[:32],
            associated_data
        )
        
        result = base64.b64encode(
            encrypted["iv"] + encrypted["tag"] + encrypted["ciphertext"]
        ).decode('ascii')
        
        return f"enc:{result}"
    
    def decrypt_field(self, encrypted_value: str, field_name: str, tenant_id: str) -> str:
        """Déchiffre un champ spécifique"""
        if not encrypted_value.startswith("enc:"):
            return encrypted_value  # Pas chiffré
        
        encrypted_data = base64.b64decode(encrypted_value[4:])
        
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        associated_data = f"field:{field_name}:tenant:{tenant_id}".encode('utf-8')
        
        encrypted_dict = {
            "ciphertext": ciphertext,
            "iv": iv,
            "tag": tag,
            "associated_data": associated_data
        }
        
        decrypted_bytes = self.decrypt_with_aes_gcm(encrypted_dict, self.master_key[:32])
        return decrypted_bytes.decode('utf-8')


class JWTManager:
    """
    Gestionnaire de tokens JWT avec rotation des clés
    """
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.redis: Optional[aioredis.Redis] = None
    
    def set_redis(self, redis_client: aioredis.Redis):
        """Configure le client Redis"""
        self.redis = redis_client
    
    def create_access_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Crée un token d'accès JWT"""
        now = datetime.utcnow()
        
        token_payload = {
            **payload,
            "iat": now,
            "exp": now + timedelta(seconds=expires_in),
            "jti": str(uuid.uuid4())  # JWT ID unique
        }
        
        return jwt.encode(token_payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str, tenant_id: str, expires_in: int = 86400 * 30) -> str:
        """Crée un token de rafraîchissement"""
        now = datetime.utcnow()
        
        payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "type": "refresh",
            "iat": now,
            "exp": now + timedelta(seconds=expires_in),
            "jti": str(uuid.uuid4())
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Vérifie et décode un token JWT"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Vérification de la révocation
            if self.redis:
                jti = payload.get("jti")
                if jti and asyncio.run(self._is_token_revoked(jti)):
                    return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    async def revoke_token(self, token: str) -> bool:
        """Révoque un token"""
        if not self.redis:
            return False
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Permet de révoquer les tokens expirés
            )
            
            jti = payload.get("jti")
            if not jti:
                return False
            
            # Calcul du TTL basé sur l'expiration
            exp = payload.get("exp")
            if exp:
                ttl = max(0, exp - datetime.utcnow().timestamp())
                await self.redis.set(f"revoked_token:{jti}", "1", ex=int(ttl))
            else:
                await self.redis.set(f"revoked_token:{jti}", "1", ex=86400)  # 24h par défaut
            
            return True
            
        except jwt.InvalidTokenError:
            return False
    
    async def _is_token_revoked(self, jti: str) -> bool:
        """Vérifie si un token est révoqué"""
        if not self.redis:
            return False
        
        result = await self.redis.get(f"revoked_token:{jti}")
        return result is not None


class IPValidator:
    """
    Validateur d'adresses IP avec support des listes noires/blanches
    """
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
    
    def is_valid_ip(self, ip_str: str) -> bool:
        """Vérifie si une chaîne est une IP valide"""
        try:
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False
    
    def is_private_ip(self, ip_str: str) -> bool:
        """Vérifie si une IP est privée"""
        try:
            ip = ipaddress.ip_address(ip_str)
            return ip.is_private
        except ValueError:
            return False
    
    def is_in_network(self, ip_str: str, network_str: str) -> bool:
        """Vérifie si une IP appartient à un réseau"""
        try:
            ip = ipaddress.ip_address(ip_str)
            network = ipaddress.ip_network(network_str, strict=False)
            return ip in network
        except ValueError:
            return False
    
    async def is_blacklisted(self, ip_str: str, tenant_id: str) -> bool:
        """Vérifie si une IP est sur liste noire"""
        # Liste noire globale
        global_blacklist = await self.redis.sismember("global_ip_blacklist", ip_str)
        if global_blacklist:
            return True
        
        # Liste noire du tenant
        tenant_blacklist = await self.redis.sismember(f"tenant:{tenant_id}:ip_blacklist", ip_str)
        return bool(tenant_blacklist)
    
    async def is_whitelisted(self, ip_str: str, tenant_id: str) -> bool:
        """Vérifie si une IP est sur liste blanche"""
        # Liste blanche globale
        global_whitelist = await self.redis.sismember("global_ip_whitelist", ip_str)
        if global_whitelist:
            return True
        
        # Liste blanche du tenant
        tenant_whitelist = await self.redis.sismember(f"tenant:{tenant_id}:ip_whitelist", ip_str)
        return bool(tenant_whitelist)
    
    async def add_to_blacklist(self, ip_str: str, tenant_id: Optional[str] = None, ttl: Optional[int] = None):
        """Ajoute une IP à la liste noire"""
        if tenant_id:
            key = f"tenant:{tenant_id}:ip_blacklist"
        else:
            key = "global_ip_blacklist"
        
        await self.redis.sadd(key, ip_str)
        
        if ttl:
            await self.redis.expire(key, ttl)
    
    async def remove_from_blacklist(self, ip_str: str, tenant_id: Optional[str] = None):
        """Retire une IP de la liste noire"""
        if tenant_id:
            key = f"tenant:{tenant_id}:ip_blacklist"
        else:
            key = "global_ip_blacklist"
        
        await self.redis.srem(key, ip_str)
    
    async def get_ip_info(self, ip_str: str) -> Dict[str, Any]:
        """Récupère les informations sur une IP"""
        info = {
            "ip": ip_str,
            "is_valid": self.is_valid_ip(ip_str),
            "is_private": False,
            "version": None,
            "is_blacklisted_global": False,
            "is_whitelisted_global": False
        }
        
        if info["is_valid"]:
            try:
                ip = ipaddress.ip_address(ip_str)
                info["is_private"] = ip.is_private
                info["version"] = ip.version
                info["is_loopback"] = ip.is_loopback
                info["is_multicast"] = ip.is_multicast
                info["is_reserved"] = ip.is_reserved
                
                # Vérification des listes
                info["is_blacklisted_global"] = await self.redis.sismember("global_ip_blacklist", ip_str)
                info["is_whitelisted_global"] = await self.redis.sismember("global_ip_whitelist", ip_str)
                
            except ValueError:
                pass
        
        return info


class SecurityAuditor:
    """
    Auditeur de sécurité pour validation des configurations
    """
    
    def __init__(self):
        self.checks = []
        
    def audit_password_policy(self, policy: Dict[str, Any]) -> List[str]:
        """Audite une politique de mot de passe"""
        issues = []
        
        min_length = policy.get("min_length", 0)
        if min_length < 8:
            issues.append("Minimum password length should be at least 8 characters")
        
        max_length = policy.get("max_length", 0)
        if max_length > 0 and max_length < 12:
            issues.append("Maximum password length seems too restrictive")
        
        if not policy.get("require_uppercase", False):
            issues.append("Password policy should require uppercase letters")
        
        if not policy.get("require_lowercase", False):
            issues.append("Password policy should require lowercase letters")
        
        if not policy.get("require_digits", False):
            issues.append("Password policy should require digits")
        
        if not policy.get("require_special_chars", False):
            issues.append("Password policy should require special characters")
        
        expiry_days = policy.get("expiry_days", 0)
        if expiry_days > 0 and expiry_days > 365:
            issues.append("Password expiry period is too long")
        
        return issues
    
    def audit_encryption_config(self, config: Dict[str, Any]) -> List[str]:
        """Audite une configuration de chiffrement"""
        issues = []
        
        algorithm = config.get("algorithm", "")
        if algorithm not in ["AES-256-GCM", "ChaCha20-Poly1305"]:
            issues.append("Weak or unsupported encryption algorithm")
        
        key_rotation = config.get("key_rotation_days", 0)
        if key_rotation == 0 or key_rotation > 90:
            issues.append("Key rotation period should be between 1-90 days")
        
        if not config.get("use_hkdf", False):
            issues.append("Should use HKDF for key derivation")
        
        return issues
    
    def audit_session_config(self, config: Dict[str, Any]) -> List[str]:
        """Audite une configuration de session"""
        issues = []
        
        session_timeout = config.get("timeout_minutes", 0)
        if session_timeout > 480:  # 8 heures
            issues.append("Session timeout is too long")
        
        if not config.get("secure_cookies", False):
            issues.append("Should use secure cookies")
        
        if not config.get("httponly_cookies", False):
            issues.append("Should use HttpOnly cookies")
        
        if not config.get("samesite_strict", False):
            issues.append("Should use SameSite strict cookies")
        
        return issues


class RateLimiter:
    """
    Limiteur de taux avec support multi-tenant
    """
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
    
    async def is_allowed(self, identifier: str, limit: int, window_seconds: int,
                        tenant_id: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Vérifie si une action est autorisée selon les limites de taux"""
        
        # Construction de la clé
        if tenant_id:
            key = f"rate_limit:tenant:{tenant_id}:{identifier}"
        else:
            key = f"rate_limit:global:{identifier}"
        
        # Implémentation sliding window avec Redis
        now = datetime.utcnow().timestamp()
        window_start = now - window_seconds
        
        # Pipeline Redis pour atomicité
        pipeline = self.redis.pipeline()
        
        # Suppression des anciennes entrées
        pipeline.zremrangebyscore(key, 0, window_start)
        
        # Ajout de la nouvelle requête
        pipeline.zadd(key, {str(uuid.uuid4()): now})
        
        # Comptage des requêtes dans la fenêtre
        pipeline.zcard(key)
        
        # Expiration de la clé
        pipeline.expire(key, window_seconds)
        
        results = await pipeline.execute()
        current_count = results[2]
        
        allowed = current_count <= limit
        
        info = {
            "allowed": allowed,
            "current_count": current_count,
            "limit": limit,
            "window_seconds": window_seconds,
            "reset_time": now + window_seconds if not allowed else None,
            "retry_after": window_seconds if not allowed else None
        }
        
        return allowed, info
    
    async def get_usage_stats(self, identifier: str, window_seconds: int,
                             tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Récupère les statistiques d'utilisation"""
        
        if tenant_id:
            key = f"rate_limit:tenant:{tenant_id}:{identifier}"
        else:
            key = f"rate_limit:global:{identifier}"
        
        now = datetime.utcnow().timestamp()
        window_start = now - window_seconds
        
        # Nettoyage et comptage
        await self.redis.zremrangebyscore(key, 0, window_start)
        current_count = await self.redis.zcard(key)
        
        # Récupération des timestamps pour analyse
        timestamps = await self.redis.zrange(key, 0, -1, withscores=True)
        
        stats = {
            "current_count": current_count,
            "window_seconds": window_seconds,
            "requests_per_minute": (current_count / window_seconds) * 60 if window_seconds > 0 else 0,
            "first_request": min([score for _, score in timestamps]) if timestamps else None,
            "last_request": max([score for _, score in timestamps]) if timestamps else None
        }
        
        return stats


class SecurityUtils:
    """
    Utilitaires de sécurité génériques
    """
    
    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """Comparaison en temps constant pour éviter les attaques temporelles"""
        return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))
    
    @staticmethod
    def generate_csrf_token() -> str:
        """Génère un token CSRF"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def hash_with_salt(data: str, salt: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Hache des données avec un salt"""
        data_bytes = data.encode('utf-8') + salt
        
        if algorithm == HashAlgorithm.SHA256:
            hash_obj = hashlib.sha256(data_bytes)
        elif algorithm == HashAlgorithm.SHA512:
            hash_obj = hashlib.sha512(data_bytes)
        elif algorithm == HashAlgorithm.SHA3_256:
            hash_obj = hashlib.sha3_256(data_bytes)
        elif algorithm == HashAlgorithm.SHA3_512:
            hash_obj = hashlib.sha3_512(data_bytes)
        elif algorithm == HashAlgorithm.BLAKE2B:
            hash_obj = hashlib.blake2b(data_bytes)
        elif algorithm == HashAlgorithm.BLAKE2S:
            hash_obj = hashlib.blake2s(data_bytes)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def generate_hmac_signature(data: str, secret: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Génère une signature HMAC"""
        if algorithm == HashAlgorithm.SHA256:
            hash_func = hashlib.sha256
        elif algorithm == HashAlgorithm.SHA512:
            hash_func = hashlib.sha512
        else:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
        
        signature = hmac.new(
            secret.encode('utf-8'),
            data.encode('utf-8'),
            hash_func
        ).hexdigest()
        
        return signature
    
    @staticmethod
    def verify_hmac_signature(data: str, secret: str, signature: str, 
                             algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bool:
        """Vérifie une signature HMAC"""
        expected_signature = SecurityUtils.generate_hmac_signature(data, secret, algorithm)
        return SecurityUtils.constant_time_compare(signature, expected_signature)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Assainit un nom de fichier pour éviter les path traversal"""
        # Suppression des caractères dangereux
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Suppression des points en début et fin
        sanitized = sanitized.strip('.')
        
        # Limitation de la longueur
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        # Éviter les noms réservés Windows
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        if sanitized.upper() in reserved_names:
            sanitized = f"_{sanitized}"
        
        return sanitized or "unnamed"
    
    @staticmethod
    def escape_html(text: str) -> str:
        """Échappe le HTML pour éviter les XSS"""
        escape_chars = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;'
        }
        
        for char, escaped in escape_chars.items():
            text = text.replace(char, escaped)
        
        return text
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Valide une adresse email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def mask_sensitive_data(data: str, mask_char: str = '*', visible_chars: int = 4) -> str:
        """Masque les données sensibles"""
        if len(data) <= visible_chars * 2:
            return mask_char * len(data)
        
        start = data[:visible_chars]
        end = data[-visible_chars:]
        middle = mask_char * (len(data) - visible_chars * 2)
        
        return start + middle + end
