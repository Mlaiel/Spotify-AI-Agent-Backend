#!/usr/bin/env python3
"""
Advanced Encryption Utilities for PagerDuty Integration

Module de chiffrement avancé pour sécuriser les données sensibles,
configurations, et communications PagerDuty. Fournit des fonctionnalités
de chiffrement robustes avec gestion des clés et audit de sécurité.

Fonctionnalités:
- Chiffrement symétrique et asymétrique
- Gestion sécurisée des clés
- Hachage et signatures
- Chiffrement de fichiers et données
- Rotation automatique des clés
- Audit de sécurité
- Conformité aux standards

Version: 1.0.0
Auteur: Spotify AI Agent Team
"""

import os
import json
import base64
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import hmac

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

class EncryptionError(Exception):
    """Exception personnalisée pour les erreurs de chiffrement"""
    pass

class KeyManager:
    """Gestionnaire de clés de chiffrement"""
    
    def __init__(self, key_store_path: Optional[str] = None):
        self.key_store_path = Path(key_store_path) if key_store_path else Path("./keys")
        self.key_store_path.mkdir(parents=True, exist_ok=True)
        self.keys = {}
        self.metadata = {}
        self._load_keys()
    
    def _load_keys(self):
        """Charge les clés depuis le store"""
        metadata_file = self.key_store_path / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                
                for key_id, meta in self.metadata.items():
                    key_file = self.key_store_path / f"{key_id}.key"
                    if key_file.exists():
                        with open(key_file, 'rb') as f:
                            self.keys[key_id] = f.read()
            except Exception as e:
                raise EncryptionError(f"Failed to load keys: {e}")
    
    def _save_metadata(self):
        """Sauvegarde les métadonnées des clés"""
        metadata_file = self.key_store_path / "metadata.json"
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            raise EncryptionError(f"Failed to save metadata: {e}")
    
    def generate_key(
        self,
        key_id: str,
        key_type: str = "fernet",
        key_size: int = 32,
        description: str = "",
        expires_days: Optional[int] = None
    ) -> str:
        """Génère une nouvelle clé de chiffrement"""
        
        if key_id in self.keys:
            raise EncryptionError(f"Key {key_id} already exists")
        
        if key_type == "fernet":
            key = Fernet.generate_key()
        elif key_type == "aes":
            key = secrets.token_bytes(key_size)
        elif key_type == "hmac":
            key = secrets.token_bytes(key_size)
        else:
            raise EncryptionError(f"Unsupported key type: {key_type}")
        
        # Sauvegarder la clé
        key_file = self.key_store_path / f"{key_id}.key"
        with open(key_file, 'wb') as f:
            f.write(key)
        
        # Mettre à jour les métadonnées
        self.metadata[key_id] = {
            "type": key_type,
            "size": len(key),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=expires_days)).isoformat() if expires_days else None,
            "usage_count": 0
        }
        
        self.keys[key_id] = key
        self._save_metadata()
        
        return base64.b64encode(key).decode()
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        """Récupère une clé par son ID"""
        if key_id not in self.keys:
            return None
        
        # Vérifier l'expiration
        meta = self.metadata.get(key_id, {})
        expires_at = meta.get("expires_at")
        
        if expires_at:
            expiry_date = datetime.fromisoformat(expires_at)
            if datetime.now(timezone.utc) > expiry_date:
                raise EncryptionError(f"Key {key_id} has expired")
        
        # Incrémenter le compteur d'usage
        self.metadata[key_id]["usage_count"] = meta.get("usage_count", 0) + 1
        self._save_metadata()
        
        return self.keys[key_id]
    
    def rotate_key(self, key_id: str, backup_old: bool = True) -> str:
        """Effectue la rotation d'une clé"""
        if key_id not in self.keys:
            raise EncryptionError(f"Key {key_id} not found")
        
        old_meta = self.metadata[key_id].copy()
        
        # Sauvegarder l'ancienne clé si demandé
        if backup_old:
            backup_id = f"{key_id}_backup_{int(datetime.now().timestamp())}"
            backup_file = self.key_store_path / f"{backup_id}.key"
            
            with open(backup_file, 'wb') as f:
                f.write(self.keys[key_id])
            
            self.metadata[backup_id] = {
                **old_meta,
                "is_backup": True,
                "original_key_id": key_id,
                "backed_up_at": datetime.now(timezone.utc).isoformat()
            }
        
        # Générer la nouvelle clé
        return self.generate_key(
            key_id + "_new",
            old_meta.get("type", "fernet"),
            old_meta.get("size", 32),
            f"Rotated key from {key_id}"
        )
    
    def delete_key(self, key_id: str, force: bool = False):
        """Supprime une clé"""
        if key_id not in self.keys:
            return
        
        meta = self.metadata.get(key_id, {})
        usage_count = meta.get("usage_count", 0)
        
        if usage_count > 0 and not force:
            raise EncryptionError(f"Key {key_id} has been used {usage_count} times. Use force=True to delete.")
        
        # Supprimer les fichiers
        key_file = self.key_store_path / f"{key_id}.key"
        if key_file.exists():
            key_file.unlink()
        
        # Supprimer des dictionnaires
        del self.keys[key_id]
        del self.metadata[key_id]
        
        self._save_metadata()
    
    def list_keys(self) -> Dict[str, Dict[str, Any]]:
        """Liste toutes les clés avec leurs métadonnées"""
        return self.metadata.copy()
    
    def cleanup_expired_keys(self) -> List[str]:
        """Nettoie les clés expirées"""
        expired_keys = []
        now = datetime.now(timezone.utc)
        
        for key_id, meta in list(self.metadata.items()):
            expires_at = meta.get("expires_at")
            if expires_at:
                expiry_date = datetime.fromisoformat(expires_at)
                if now > expiry_date:
                    self.delete_key(key_id, force=True)
                    expired_keys.append(key_id)
        
        return expired_keys

class SymmetricEncryption:
    """Chiffrement symétrique avec Fernet"""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
    
    def encrypt(self, data: Union[str, bytes], key_id: str) -> str:
        """Chiffre des données"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise EncryptionError("Cryptography library not available")
        
        key = self.key_manager.get_key(key_id)
        if not key:
            raise EncryptionError(f"Key {key_id} not found")
        
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            fernet = Fernet(key)
            encrypted_data = fernet.encrypt(data)
            
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}")
    
    def decrypt(self, encrypted_data: str, key_id: str) -> str:
        """Déchiffre des données"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise EncryptionError("Cryptography library not available")
        
        key = self.key_manager.get_key(key_id)
        if not key:
            raise EncryptionError(f"Key {key_id} not found")
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_bytes)
            
            return decrypted_data.decode('utf-8')
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}")
    
    def encrypt_json(self, data: Dict[str, Any], key_id: str) -> str:
        """Chiffre des données JSON"""
        json_str = json.dumps(data, separators=(',', ':'))
        return self.encrypt(json_str, key_id)
    
    def decrypt_json(self, encrypted_data: str, key_id: str) -> Dict[str, Any]:
        """Déchiffre des données JSON"""
        json_str = self.decrypt(encrypted_data, key_id)
        return json.loads(json_str)

class AsymmetricEncryption:
    """Chiffrement asymétrique avec RSA"""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
    
    def generate_key_pair(self, key_id: str, key_size: int = 2048) -> Tuple[str, str]:
        """Génère une paire de clés RSA"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise EncryptionError("Cryptography library not available")
        
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            
            public_key = private_key.public_key()
            
            # Sérialiser les clés
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Sauvegarder les clés
            private_file = self.key_manager.key_store_path / f"{key_id}_private.pem"
            public_file = self.key_manager.key_store_path / f"{key_id}_public.pem"
            
            with open(private_file, 'wb') as f:
                f.write(private_pem)
            
            with open(public_file, 'wb') as f:
                f.write(public_pem)
            
            # Mettre à jour les métadonnées
            self.key_manager.metadata[f"{key_id}_keypair"] = {
                "type": "rsa",
                "size": key_size,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "private_key_file": str(private_file),
                "public_key_file": str(public_file)
            }
            
            self.key_manager._save_metadata()
            
            return (
                base64.b64encode(private_pem).decode(),
                base64.b64encode(public_pem).decode()
            )
            
        except Exception as e:
            raise EncryptionError(f"Key pair generation failed: {e}")
    
    def encrypt_with_public_key(self, data: Union[str, bytes], public_key_pem: str) -> str:
        """Chiffre avec une clé publique"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise EncryptionError("Cryptography library not available")
        
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            public_key_bytes = base64.b64decode(public_key_pem.encode())
            public_key = serialization.load_pem_public_key(
                public_key_bytes,
                backend=default_backend()
            )
            
            encrypted_data = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            raise EncryptionError(f"Public key encryption failed: {e}")
    
    def decrypt_with_private_key(self, encrypted_data: str, private_key_pem: str) -> str:
        """Déchiffre avec une clé privée"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise EncryptionError("Cryptography library not available")
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            private_key_bytes = base64.b64decode(private_key_pem.encode())
            
            private_key = serialization.load_pem_private_key(
                private_key_bytes,
                password=None,
                backend=default_backend()
            )
            
            decrypted_data = private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            raise EncryptionError(f"Private key decryption failed: {e}")

class HashingUtility:
    """Utilitaires de hachage et vérification d'intégrité"""
    
    @staticmethod
    def hash_data(
        data: Union[str, bytes],
        algorithm: str = "sha256",
        salt: Optional[bytes] = None
    ) -> str:
        """Calcule le hash de données"""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if salt:
            data = salt + data
        
        if algorithm == "sha256":
            hash_obj = hashlib.sha256()
        elif algorithm == "sha512":
            hash_obj = hashlib.sha512()
        elif algorithm == "md5":
            hash_obj = hashlib.md5()
        else:
            raise EncryptionError(f"Unsupported hash algorithm: {algorithm}")
        
        hash_obj.update(data)
        return hash_obj.hexdigest()
    
    @staticmethod
    def generate_salt(length: int = 32) -> bytes:
        """Génère un sel aléatoire"""
        return secrets.token_bytes(length)
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        """Hache un mot de passe avec sel"""
        if salt is None:
            salt = HashingUtility.generate_salt()
        
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback simple
            return HashingUtility.hash_data(password, "sha256", salt), salt
        
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            
            key = kdf.derive(password.encode('utf-8'))
            return base64.b64encode(key).decode(), salt
            
        except Exception as e:
            raise EncryptionError(f"Password hashing failed: {e}")
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: bytes) -> bool:
        """Vérifie un mot de passe"""
        try:
            new_hash, _ = HashingUtility.hash_password(password, salt)
            return hmac.compare_digest(hashed, new_hash)
        except Exception:
            return False
    
    @staticmethod
    def create_hmac_signature(
        data: Union[str, bytes],
        secret_key: bytes,
        algorithm: str = "sha256"
    ) -> str:
        """Crée une signature HMAC"""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == "sha256":
            mac = hmac.new(secret_key, data, hashlib.sha256)
        elif algorithm == "sha512":
            mac = hmac.new(secret_key, data, hashlib.sha512)
        else:
            raise EncryptionError(f"Unsupported HMAC algorithm: {algorithm}")
        
        return mac.hexdigest()
    
    @staticmethod
    def verify_hmac_signature(
        data: Union[str, bytes],
        signature: str,
        secret_key: bytes,
        algorithm: str = "sha256"
    ) -> bool:
        """Vérifie une signature HMAC"""
        try:
            expected_signature = HashingUtility.create_hmac_signature(
                data, secret_key, algorithm
            )
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False

class FileEncryption:
    """Chiffrement de fichiers"""
    
    def __init__(self, symmetric_encryption: SymmetricEncryption):
        self.encryption = symmetric_encryption
    
    def encrypt_file(
        self,
        input_file: str,
        output_file: str,
        key_id: str,
        chunk_size: int = 8192
    ) -> bool:
        """Chiffre un fichier"""
        try:
            with open(input_file, 'rb') as infile:
                data = infile.read()
            
            encrypted_data = self.encryption.encrypt(data, key_id)
            
            with open(output_file, 'w') as outfile:
                outfile.write(encrypted_data)
            
            return True
            
        except Exception as e:
            raise EncryptionError(f"File encryption failed: {e}")
    
    def decrypt_file(
        self,
        input_file: str,
        output_file: str,
        key_id: str
    ) -> bool:
        """Déchiffre un fichier"""
        try:
            with open(input_file, 'r') as infile:
                encrypted_data = infile.read()
            
            decrypted_data = self.encryption.decrypt(encrypted_data, key_id)
            
            with open(output_file, 'wb') as outfile:
                outfile.write(decrypted_data.encode('utf-8'))
            
            return True
            
        except Exception as e:
            raise EncryptionError(f"File decryption failed: {e}")

class SecureStorage:
    """Stockage sécurisé de données sensibles"""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self.encryption = SymmetricEncryption(key_manager)
        self.storage = {}
    
    def store_secret(
        self,
        secret_name: str,
        secret_value: str,
        key_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Stocke un secret de manière sécurisée"""
        try:
            if key_id is None:
                key_id = f"secret_{secret_name}"
                if key_id not in self.key_manager.keys:
                    self.key_manager.generate_key(key_id, description=f"Key for secret {secret_name}")
            
            encrypted_value = self.encryption.encrypt(secret_value, key_id)
            
            self.storage[secret_name] = {
                "encrypted_value": encrypted_value,
                "key_id": key_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {}
            }
            
            return True
            
        except Exception as e:
            raise EncryptionError(f"Failed to store secret: {e}")
    
    def retrieve_secret(self, secret_name: str) -> Optional[str]:
        """Récupère un secret"""
        if secret_name not in self.storage:
            return None
        
        try:
            secret_data = self.storage[secret_name]
            return self.encryption.decrypt(
                secret_data["encrypted_value"],
                secret_data["key_id"]
            )
        except Exception as e:
            raise EncryptionError(f"Failed to retrieve secret: {e}")
    
    def delete_secret(self, secret_name: str) -> bool:
        """Supprime un secret"""
        if secret_name in self.storage:
            del self.storage[secret_name]
            return True
        return False
    
    def list_secrets(self) -> List[str]:
        """Liste les noms des secrets stockés"""
        return list(self.storage.keys())

# Fonctions utilitaires
def generate_random_key(length: int = 32) -> str:
    """Génère une clé aléatoire"""
    return base64.b64encode(secrets.token_bytes(length)).decode()

def secure_compare(a: str, b: str) -> bool:
    """Compare deux chaînes de manière sécurisée"""
    return hmac.compare_digest(a, b)

def is_strong_password(password: str, min_length: int = 12) -> bool:
    """Vérifie si un mot de passe est fort"""
    if len(password) < min_length:
        return False
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in password)
    
    return all([has_upper, has_lower, has_digit, has_special])

# Export des classes principales
__all__ = [
    "EncryptionError",
    "KeyManager",
    "SymmetricEncryption",
    "AsymmetricEncryption", 
    "HashingUtility",
    "FileEncryption",
    "SecureStorage",
    "generate_random_key",
    "secure_compare",
    "is_strong_password"
]
