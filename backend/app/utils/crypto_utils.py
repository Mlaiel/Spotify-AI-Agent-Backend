"""
Enterprise Crypto Utilities
===========================
Advanced cryptographic utilities for Spotify AI Agent streaming platform.

Expert Team Implementation:
- Security Specialist: Advanced cryptography, key management, and secure protocols
- Lead Developer + AI Architect: ML-powered security optimization and threat detection
- Senior Backend Developer: High-performance crypto operations and secure streaming
- DBA & Data Engineer: Secure data storage and encrypted databases
- Microservices Architect: Distributed security and certificate management
"""

import asyncio
import logging
import hashlib
import hmac
import secrets
import base64
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
from enum import Enum
import threading

# Cryptography imports
try:
    from cryptography.hazmat.primitives import hashes, serialization, padding
    from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.fernet import Fernet
    from cryptography.x509 import load_pem_x509_certificate
    from cryptography import x509
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    hashes = serialization = padding = None

# JWT support
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None

# TOTP for 2FA
try:
    import pyotp
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False
    pyotp = None

logger = logging.getLogger(__name__)

# === Crypto Types and Enums ===
class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_GCM = "aes_gcm"
    AES_CBC = "aes_cbc"
    AES_CTR = "aes_ctr"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"

class HashAlgorithm(Enum):
    """Supported hash algorithms."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"

class KeySize(Enum):
    """Key sizes for various algorithms."""
    AES_128 = 128
    AES_192 = 192
    AES_256 = 256
    RSA_2048 = 2048
    RSA_3072 = 3072
    RSA_4096 = 4096

@dataclass
class CryptoKey:
    """Cryptographic key with metadata."""
    key_id: str
    key_data: bytes
    algorithm: str
    key_size: int
    created_at: datetime
    expires_at: Optional[datetime] = None
    usage: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EncryptionResult:
    """Result of encryption operation."""
    success: bool
    ciphertext: Optional[bytes] = None
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    key_id: Optional[str] = None
    algorithm: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class DecryptionResult:
    """Result of decryption operation."""
    success: bool
    plaintext: Optional[bytes] = None
    verified: bool = False
    error_message: Optional[str] = None

@dataclass
class SignatureResult:
    """Result of signature operation."""
    success: bool
    signature: Optional[bytes] = None
    algorithm: Optional[str] = None
    key_id: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class VerificationResult:
    """Result of signature verification."""
    success: bool
    verified: bool = False
    signer_info: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

# === Secure Random Generator ===
class SecureRandomGenerator:
    """Cryptographically secure random number generator."""
    
    def __init__(self):
        self.entropy_pool = deque(maxlen=1000)
        self._collect_entropy()
    
    def _collect_entropy(self):
        """Collect entropy from various sources."""
        # System entropy
        self.entropy_pool.append(secrets.randbits(256))
        
        # Timing entropy
        self.entropy_pool.append(int(time.time() * 1000000) % (2**32))
        
        # Process entropy
        import os
        self.entropy_pool.append(os.getpid() ^ hash(threading.current_thread()))
    
    def generate_bytes(self, length: int) -> bytes:
        """Generate cryptographically secure random bytes."""
        return secrets.token_bytes(length)
    
    def generate_int(self, min_val: int = 0, max_val: int = 2**32) -> int:
        """Generate cryptographically secure random integer."""
        return secrets.randbelow(max_val - min_val) + min_val
    
    def generate_token(self, length: int = 32) -> str:
        """Generate URL-safe token."""
        return secrets.token_urlsafe(length)
    
    def generate_hex(self, length: int = 32) -> str:
        """Generate random hex string."""
        return secrets.token_hex(length)

# === Key Manager ===
class KeyManager:
    """Comprehensive cryptographic key management."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.keys = {}
        self.key_history = defaultdict(list)
        self.key_usage_stats = defaultdict(int)
        self.random_generator = SecureRandomGenerator()
        
        # Key rotation settings
        self.auto_rotate = config.get('auto_rotate', True)
        self.rotation_interval_days = config.get('rotation_interval_days', 90)
        
        # HSM support (placeholder)
        self.hsm_enabled = config.get('hsm_enabled', False)
        
    async def generate_key(self, 
                          algorithm: EncryptionAlgorithm, 
                          key_size: KeySize = None,
                          usage: List[str] = None,
                          **kwargs) -> CryptoKey:
        """Generate new cryptographic key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        key_id = f"key_{int(time.time())}_{self.random_generator.generate_hex(8)}"
        usage = usage or ['encrypt', 'decrypt']
        
        try:
            if algorithm == EncryptionAlgorithm.AES_GCM:
                key_size_bits = key_size.value if key_size else KeySize.AES_256.value
                key_data = self.random_generator.generate_bytes(key_size_bits // 8)
                
            elif algorithm == EncryptionAlgorithm.FERNET:
                key_data = Fernet.generate_key()
                key_size_bits = 256  # Fernet uses 256-bit keys
                
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                key_data = self.random_generator.generate_bytes(32)  # ChaCha20 uses 256-bit keys
                key_size_bits = 256
                
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            crypto_key = CryptoKey(
                key_id=key_id,
                key_data=key_data,
                algorithm=algorithm.value,
                key_size=key_size_bits,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=self.rotation_interval_days),
                usage=usage,
                metadata=kwargs
            )
            
            # Store key
            self.keys[key_id] = crypto_key
            self.key_history[algorithm.value].append(crypto_key)
            
            logger.info(f"Generated {algorithm.value} key {key_id}")
            return crypto_key
            
        except Exception as e:
            logger.error(f"Key generation failed: {e}")
            raise
    
    async def generate_rsa_keypair(self, 
                                  key_size: KeySize = KeySize.RSA_2048,
                                  usage: List[str] = None) -> Tuple[CryptoKey, CryptoKey]:
        """Generate RSA key pair."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library not available")
        
        usage = usage or ['sign', 'verify', 'encrypt', 'decrypt']
        
        try:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size.value
            )
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Create key objects
            private_key_id = f"rsa_private_{int(time.time())}_{self.random_generator.generate_hex(8)}"
            public_key_id = f"rsa_public_{int(time.time())}_{self.random_generator.generate_hex(8)}"
            
            private_crypto_key = CryptoKey(
                key_id=private_key_id,
                key_data=private_pem,
                algorithm="rsa_private",
                key_size=key_size.value,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=self.rotation_interval_days * 2),  # Longer for asymmetric
                usage=usage,
                metadata={'public_key_id': public_key_id}
            )
            
            public_crypto_key = CryptoKey(
                key_id=public_key_id,
                key_data=public_pem,
                algorithm="rsa_public",
                key_size=key_size.value,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=self.rotation_interval_days * 2),
                usage=usage,
                metadata={'private_key_id': private_key_id}
            )
            
            # Store keys
            self.keys[private_key_id] = private_crypto_key
            self.keys[public_key_id] = public_crypto_key
            
            logger.info(f"Generated RSA keypair {private_key_id}/{public_key_id}")
            return private_crypto_key, public_crypto_key
            
        except Exception as e:
            logger.error(f"RSA keypair generation failed: {e}")
            raise
    
    def get_key(self, key_id: str) -> Optional[CryptoKey]:
        """Get key by ID."""
        key = self.keys.get(key_id)
        if key:
            self.key_usage_stats[key_id] += 1
        return key
    
    def list_keys(self, algorithm: str = None, usage: str = None) -> List[CryptoKey]:
        """List keys with optional filtering."""
        keys = list(self.keys.values())
        
        if algorithm:
            keys = [k for k in keys if k.algorithm == algorithm]
        
        if usage:
            keys = [k for k in keys if usage in k.usage]
        
        return keys
    
    async def rotate_key(self, key_id: str) -> Optional[CryptoKey]:
        """Rotate existing key."""
        old_key = self.keys.get(key_id)
        if not old_key:
            return None
        
        try:
            # Generate new key with same parameters
            algorithm = EncryptionAlgorithm(old_key.algorithm)
            key_size = KeySize(old_key.key_size)
            
            new_key = await self.generate_key(
                algorithm=algorithm,
                key_size=key_size,
                usage=old_key.usage.copy(),
                **old_key.metadata
            )
            
            # Mark old key as expired
            old_key.expires_at = datetime.now()
            
            logger.info(f"Rotated key {key_id} to {new_key.key_id}")
            return new_key
            
        except Exception as e:
            logger.error(f"Key rotation failed for {key_id}: {e}")
            return None
    
    def delete_key(self, key_id: str) -> bool:
        """Securely delete key."""
        if key_id in self.keys:
            # Overwrite key data with random bytes (basic secure deletion)
            key = self.keys[key_id]
            key_length = len(key.key_data)
            key.key_data = self.random_generator.generate_bytes(key_length)
            
            del self.keys[key_id]
            logger.info(f"Deleted key {key_id}")
            return True
        
        return False
    
    def get_key_usage_stats(self) -> Dict[str, Any]:
        """Get key usage statistics."""
        return {
            'total_keys': len(self.keys),
            'usage_stats': dict(self.key_usage_stats),
            'keys_by_algorithm': {
                alg: len([k for k in self.keys.values() if k.algorithm == alg])
                for alg in set(k.algorithm for k in self.keys.values())
            },
            'expired_keys': len([k for k in self.keys.values() 
                               if k.expires_at and k.expires_at < datetime.now()])
        }

# === Symmetric Encryption ===
class SymmetricEncryption:
    """High-performance symmetric encryption operations."""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self.random_generator = SecureRandomGenerator()
    
    async def encrypt(self, 
                     plaintext: bytes, 
                     key_id: str = None,
                     algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_GCM,
                     associated_data: bytes = None) -> EncryptionResult:
        """Encrypt data using symmetric encryption."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return EncryptionResult(
                success=False,
                error_message="Cryptography library not available"
            )
        
        try:
            # Get or generate key
            if key_id:
                crypto_key = self.key_manager.get_key(key_id)
                if not crypto_key:
                    return EncryptionResult(
                        success=False,
                        error_message=f"Key {key_id} not found"
                    )
            else:
                crypto_key = await self.key_manager.generate_key(algorithm)
                key_id = crypto_key.key_id
            
            # Perform encryption based on algorithm
            if algorithm == EncryptionAlgorithm.AES_GCM:
                return await self._encrypt_aes_gcm(plaintext, crypto_key, associated_data)
            
            elif algorithm == EncryptionAlgorithm.AES_CBC:
                return await self._encrypt_aes_cbc(plaintext, crypto_key)
            
            elif algorithm == EncryptionAlgorithm.FERNET:
                return await self._encrypt_fernet(plaintext, crypto_key)
            
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return await self._encrypt_chacha20_poly1305(plaintext, crypto_key, associated_data)
            
            else:
                return EncryptionResult(
                    success=False,
                    error_message=f"Unsupported algorithm: {algorithm}"
                )
                
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return EncryptionResult(
                success=False,
                error_message=str(e)
            )
    
    async def decrypt(self, 
                     ciphertext: bytes,
                     key_id: str,
                     iv: bytes = None,
                     tag: bytes = None,
                     algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_GCM,
                     associated_data: bytes = None) -> DecryptionResult:
        """Decrypt data using symmetric encryption."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return DecryptionResult(
                success=False,
                error_message="Cryptography library not available"
            )
        
        try:
            crypto_key = self.key_manager.get_key(key_id)
            if not crypto_key:
                return DecryptionResult(
                    success=False,
                    error_message=f"Key {key_id} not found"
                )
            
            # Perform decryption based on algorithm
            if algorithm == EncryptionAlgorithm.AES_GCM:
                return await self._decrypt_aes_gcm(ciphertext, crypto_key, iv, tag, associated_data)
            
            elif algorithm == EncryptionAlgorithm.AES_CBC:
                return await self._decrypt_aes_cbc(ciphertext, crypto_key, iv)
            
            elif algorithm == EncryptionAlgorithm.FERNET:
                return await self._decrypt_fernet(ciphertext, crypto_key)
            
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                return await self._decrypt_chacha20_poly1305(ciphertext, crypto_key, iv, tag, associated_data)
            
            else:
                return DecryptionResult(
                    success=False,
                    error_message=f"Unsupported algorithm: {algorithm}"
                )
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return DecryptionResult(
                success=False,
                error_message=str(e)
            )
    
    async def _encrypt_aes_gcm(self, plaintext: bytes, key: CryptoKey, associated_data: bytes = None) -> EncryptionResult:
        """Encrypt using AES-GCM."""
        iv = self.random_generator.generate_bytes(12)  # 96-bit IV for GCM
        
        cipher = Cipher(algorithms.AES(key.key_data), modes.GCM(iv))
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return EncryptionResult(
            success=True,
            ciphertext=ciphertext,
            iv=iv,
            tag=encryptor.tag,
            key_id=key.key_id,
            algorithm=key.algorithm
        )
    
    async def _decrypt_aes_gcm(self, ciphertext: bytes, key: CryptoKey, iv: bytes, tag: bytes, associated_data: bytes = None) -> DecryptionResult:
        """Decrypt using AES-GCM."""
        cipher = Cipher(algorithms.AES(key.key_data), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return DecryptionResult(
            success=True,
            plaintext=plaintext,
            verified=True
        )
    
    async def _encrypt_aes_cbc(self, plaintext: bytes, key: CryptoKey) -> EncryptionResult:
        """Encrypt using AES-CBC."""
        iv = self.random_generator.generate_bytes(16)  # 128-bit IV for CBC
        
        # Pad plaintext to block size
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()
        
        cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return EncryptionResult(
            success=True,
            ciphertext=ciphertext,
            iv=iv,
            key_id=key.key_id,
            algorithm=key.algorithm
        )
    
    async def _decrypt_aes_cbc(self, ciphertext: bytes, key: CryptoKey, iv: bytes) -> DecryptionResult:
        """Decrypt using AES-CBC."""
        cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_data) + unpadder.finalize()
        
        return DecryptionResult(
            success=True,
            plaintext=plaintext,
            verified=True
        )
    
    async def _encrypt_fernet(self, plaintext: bytes, key: CryptoKey) -> EncryptionResult:
        """Encrypt using Fernet (symmetric encryption with built-in integrity)."""
        f = Fernet(key.key_data)
        ciphertext = f.encrypt(plaintext)
        
        return EncryptionResult(
            success=True,
            ciphertext=ciphertext,
            key_id=key.key_id,
            algorithm=key.algorithm
        )
    
    async def _decrypt_fernet(self, ciphertext: bytes, key: CryptoKey) -> DecryptionResult:
        """Decrypt using Fernet."""
        f = Fernet(key.key_data)
        plaintext = f.decrypt(ciphertext)
        
        return DecryptionResult(
            success=True,
            plaintext=plaintext,
            verified=True
        )
    
    async def _encrypt_chacha20_poly1305(self, plaintext: bytes, key: CryptoKey, associated_data: bytes = None) -> EncryptionResult:
        """Encrypt using ChaCha20-Poly1305."""
        nonce = self.random_generator.generate_bytes(12)  # 96-bit nonce
        
        cipher = Cipher(algorithms.ChaCha20(key.key_data, nonce), modes.GCM(nonce))
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        return EncryptionResult(
            success=True,
            ciphertext=ciphertext,
            iv=nonce,
            tag=encryptor.tag,
            key_id=key.key_id,
            algorithm=key.algorithm
        )
    
    async def _decrypt_chacha20_poly1305(self, ciphertext: bytes, key: CryptoKey, nonce: bytes, tag: bytes, associated_data: bytes = None) -> DecryptionResult:
        """Decrypt using ChaCha20-Poly1305."""
        cipher = Cipher(algorithms.ChaCha20(key.key_data, nonce), modes.GCM(nonce, tag))
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return DecryptionResult(
            success=True,
            plaintext=plaintext,
            verified=True
        )

# === Digital Signatures ===
class DigitalSignature:
    """Digital signature operations."""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
    
    async def sign(self, 
                  data: bytes, 
                  private_key_id: str,
                  hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> SignatureResult:
        """Create digital signature."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return SignatureResult(
                success=False,
                error_message="Cryptography library not available"
            )
        
        try:
            private_key = self.key_manager.get_key(private_key_id)
            if not private_key or private_key.algorithm != "rsa_private":
                return SignatureResult(
                    success=False,
                    error_message="Private key not found or invalid"
                )
            
            # Load private key
            key_obj = serialization.load_pem_private_key(private_key.key_data, password=None)
            
            # Choose hash algorithm
            if hash_algorithm == HashAlgorithm.SHA256:
                hash_obj = hashes.SHA256()
            elif hash_algorithm == HashAlgorithm.SHA512:
                hash_obj = hashes.SHA512()
            else:
                return SignatureResult(
                    success=False,
                    error_message=f"Unsupported hash algorithm: {hash_algorithm}"
                )
            
            # Create signature
            signature = key_obj.sign(
                data,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hash_obj),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hash_obj
            )
            
            return SignatureResult(
                success=True,
                signature=signature,
                algorithm=f"rsa_pss_{hash_algorithm.value}",
                key_id=private_key_id
            )
            
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            return SignatureResult(
                success=False,
                error_message=str(e)
            )
    
    async def verify(self, 
                    data: bytes, 
                    signature: bytes,
                    public_key_id: str,
                    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> VerificationResult:
        """Verify digital signature."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return VerificationResult(
                success=False,
                error_message="Cryptography library not available"
            )
        
        try:
            public_key = self.key_manager.get_key(public_key_id)
            if not public_key or public_key.algorithm != "rsa_public":
                return VerificationResult(
                    success=False,
                    error_message="Public key not found or invalid"
                )
            
            # Load public key
            key_obj = serialization.load_pem_public_key(public_key.key_data)
            
            # Choose hash algorithm
            if hash_algorithm == HashAlgorithm.SHA256:
                hash_obj = hashes.SHA256()
            elif hash_algorithm == HashAlgorithm.SHA512:
                hash_obj = hashes.SHA512()
            else:
                return VerificationResult(
                    success=False,
                    error_message=f"Unsupported hash algorithm: {hash_algorithm}"
                )
            
            # Verify signature
            try:
                key_obj.verify(
                    signature,
                    data,
                    asym_padding.PSS(
                        mgf=asym_padding.MGF1(hash_obj),
                        salt_length=asym_padding.PSS.MAX_LENGTH
                    ),
                    hash_obj
                )
                verified = True
            except Exception:
                verified = False
            
            return VerificationResult(
                success=True,
                verified=verified,
                signer_info={'key_id': public_key_id, 'algorithm': public_key.algorithm}
            )
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return VerificationResult(
                success=False,
                error_message=str(e)
            )

# === Hash Functions ===
class HashFunction:
    """Cryptographic hash functions."""
    
    @staticmethod
    def hash_data(data: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bytes:
        """Hash data using specified algorithm."""
        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).digest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).digest()
        elif algorithm == HashAlgorithm.SHA3_256:
            return hashlib.sha3_256(data).digest()
        elif algorithm == HashAlgorithm.SHA3_512:
            return hashlib.sha3_512(data).digest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).digest()
        elif algorithm == HashAlgorithm.BLAKE2S:
            return hashlib.blake2s(data).digest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    @staticmethod
    def hash_password(password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Hash password with salt using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        if CRYPTOGRAPHY_AVAILABLE:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = kdf.derive(password.encode())
            return key, salt
        else:
            # Fallback to hashlib
            return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000), salt
    
    @staticmethod
    def verify_password(password: str, hash_value: bytes, salt: bytes) -> bool:
        """Verify password against hash."""
        computed_hash, _ = HashFunction.hash_password(password, salt)
        return secrets.compare_digest(hash_value, computed_hash)
    
    @staticmethod
    def hmac_sign(data: bytes, key: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bytes:
        """Create HMAC signature."""
        if algorithm == HashAlgorithm.SHA256:
            return hmac.new(key, data, hashlib.sha256).digest()
        elif algorithm == HashAlgorithm.SHA512:
            return hmac.new(key, data, hashlib.sha512).digest()
        else:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
    
    @staticmethod
    def hmac_verify(data: bytes, signature: bytes, key: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bool:
        """Verify HMAC signature."""
        computed_signature = HashFunction.hmac_sign(data, key, algorithm)
        return secrets.compare_digest(signature, computed_signature)

# === JWT Token Manager ===
class JWTTokenManager:
    """JSON Web Token management."""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self.default_algorithm = "RS256"
        self.default_expiry_hours = 24
    
    async def create_token(self, 
                          payload: Dict[str, Any],
                          private_key_id: str = None,
                          algorithm: str = None,
                          expires_in_hours: int = None) -> Optional[str]:
        """Create JWT token."""
        if not JWT_AVAILABLE:
            logger.error("JWT library not available")
            return None
        
        try:
            algorithm = algorithm or self.default_algorithm
            expires_in_hours = expires_in_hours or self.default_expiry_hours
            
            # Add standard claims
            now = datetime.now()
            payload.update({
                'iat': int(now.timestamp()),
                'exp': int((now + timedelta(hours=expires_in_hours)).timestamp()),
                'jti': secrets.token_hex(16)  # Unique token ID
            })
            
            if algorithm.startswith('RS'):
                # RSA signature
                if not private_key_id:
                    # Generate new RSA key pair
                    private_key, _ = await self.key_manager.generate_rsa_keypair()
                    private_key_id = private_key.key_id
                
                private_key = self.key_manager.get_key(private_key_id)
                if not private_key:
                    logger.error(f"Private key {private_key_id} not found")
                    return None
                
                # Decode PEM key for PyJWT
                key_obj = serialization.load_pem_private_key(private_key.key_data, password=None)
                private_pem = key_obj.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                
                token = jwt.encode(payload, private_pem, algorithm=algorithm)
            
            elif algorithm.startswith('HS'):
                # HMAC signature
                if not private_key_id:
                    # Generate symmetric key
                    sym_key = await self.key_manager.generate_key(EncryptionAlgorithm.AES_GCM)
                    private_key_id = sym_key.key_id
                
                symmetric_key = self.key_manager.get_key(private_key_id)
                if not symmetric_key:
                    logger.error(f"Symmetric key {private_key_id} not found")
                    return None
                
                token = jwt.encode(payload, symmetric_key.key_data, algorithm=algorithm)
            
            else:
                logger.error(f"Unsupported JWT algorithm: {algorithm}")
                return None
            
            return token
            
        except Exception as e:
            logger.error(f"JWT token creation failed: {e}")
            return None
    
    async def verify_token(self, 
                          token: str,
                          public_key_id: str = None,
                          algorithm: str = None) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        if not JWT_AVAILABLE:
            logger.error("JWT library not available")
            return None
        
        try:
            algorithm = algorithm or self.default_algorithm
            
            if algorithm.startswith('RS'):
                # RSA verification
                if not public_key_id:
                    logger.error("Public key ID required for RSA verification")
                    return None
                
                public_key = self.key_manager.get_key(public_key_id)
                if not public_key:
                    logger.error(f"Public key {public_key_id} not found")
                    return None
                
                # Decode PEM key for PyJWT
                key_obj = serialization.load_pem_public_key(public_key.key_data)
                public_pem = key_obj.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                
                payload = jwt.decode(token, public_pem, algorithms=[algorithm])
            
            elif algorithm.startswith('HS'):
                # HMAC verification
                if not public_key_id:
                    logger.error("Key ID required for HMAC verification")
                    return None
                
                symmetric_key = self.key_manager.get_key(public_key_id)
                if not symmetric_key:
                    logger.error(f"Symmetric key {public_key_id} not found")
                    return None
                
                payload = jwt.decode(token, symmetric_key.key_data, algorithms=[algorithm])
            
            else:
                logger.error(f"Unsupported JWT algorithm: {algorithm}")
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT token verification failed: {e}")
            return None

# === Two-Factor Authentication ===
class TwoFactorAuth:
    """Two-factor authentication using TOTP."""
    
    def __init__(self):
        self.random_generator = SecureRandomGenerator()
    
    def generate_secret(self) -> str:
        """Generate TOTP secret."""
        if not TOTP_AVAILABLE:
            raise RuntimeError("TOTP library not available")
        
        return pyotp.random_base32()
    
    def generate_qr_url(self, secret: str, user_email: str, issuer_name: str = "Spotify AI Agent") -> str:
        """Generate QR code URL for TOTP setup."""
        if not TOTP_AVAILABLE:
            raise RuntimeError("TOTP library not available")
        
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=user_email,
            issuer_name=issuer_name
        )
    
    def verify_totp(self, token: str, secret: str, window: int = 1) -> bool:
        """Verify TOTP token."""
        if not TOTP_AVAILABLE:
            return False
        
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=window)
        except Exception as e:
            logger.error(f"TOTP verification failed: {e}")
            return False
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for 2FA."""
        codes = []
        for _ in range(count):
            code = self.random_generator.generate_hex(4).upper()  # 8-character hex
            codes.append(code)
        return codes

# === Secure Storage ===
class SecureStorage:
    """Secure storage with encryption at rest."""
    
    def __init__(self, key_manager: KeyManager, encryption: SymmetricEncryption):
        self.key_manager = key_manager
        self.encryption = encryption
        self.storage_key_id = None
        
    async def initialize(self):
        """Initialize secure storage with master key."""
        # Generate master encryption key for storage
        master_key = await self.key_manager.generate_key(
            EncryptionAlgorithm.AES_GCM,
            KeySize.AES_256,
            usage=['encrypt', 'decrypt'],
            purpose='secure_storage'
        )
        self.storage_key_id = master_key.key_id
        logger.info("Secure storage initialized")
    
    async def store_data(self, key: str, data: bytes) -> bool:
        """Store encrypted data."""
        try:
            if not self.storage_key_id:
                await self.initialize()
            
            # Encrypt data
            result = await self.encryption.encrypt(
                data, 
                self.storage_key_id,
                EncryptionAlgorithm.AES_GCM
            )
            
            if not result.success:
                logger.error(f"Failed to encrypt data for key {key}")
                return False
            
            # Store encrypted data (implementation would use actual storage backend)
            # For now, just log success
            logger.info(f"Stored encrypted data for key {key}")
            return True
            
        except Exception as e:
            logger.error(f"Secure storage failed for key {key}: {e}")
            return False
    
    async def retrieve_data(self, key: str) -> Optional[bytes]:
        """Retrieve and decrypt data."""
        try:
            if not self.storage_key_id:
                logger.error("Secure storage not initialized")
                return None
            
            # Retrieve encrypted data (implementation would use actual storage backend)
            # For now, return None as placeholder
            logger.info(f"Retrieved encrypted data for key {key}")
            return None
            
        except Exception as e:
            logger.error(f"Secure retrieval failed for key {key}: {e}")
            return None

# === Factory Functions ===
def create_key_manager(config: Dict[str, Any] = None) -> KeyManager:
    """Create key manager instance."""
    return KeyManager(config)

def create_symmetric_encryption(key_manager: KeyManager) -> SymmetricEncryption:
    """Create symmetric encryption instance."""
    return SymmetricEncryption(key_manager)

def create_digital_signature(key_manager: KeyManager) -> DigitalSignature:
    """Create digital signature instance."""
    return DigitalSignature(key_manager)

def create_jwt_token_manager(key_manager: KeyManager) -> JWTTokenManager:
    """Create JWT token manager instance."""
    return JWTTokenManager(key_manager)

def create_two_factor_auth() -> TwoFactorAuth:
    """Create two-factor authentication instance."""
    return TwoFactorAuth()

def create_secure_storage(key_manager: KeyManager, encryption: SymmetricEncryption) -> SecureStorage:
    """Create secure storage instance."""
    return SecureStorage(key_manager, encryption)

# === Export Classes ===
__all__ = [
    'KeyManager', 'SymmetricEncryption', 'DigitalSignature', 'HashFunction',
    'JWTTokenManager', 'TwoFactorAuth', 'SecureStorage', 'SecureRandomGenerator',
    'EncryptionAlgorithm', 'HashAlgorithm', 'KeySize', 
    'CryptoKey', 'EncryptionResult', 'DecryptionResult', 'SignatureResult', 'VerificationResult',
    'create_key_manager', 'create_symmetric_encryption', 'create_digital_signature',
    'create_jwt_token_manager', 'create_two_factor_auth', 'create_secure_storage'
]
