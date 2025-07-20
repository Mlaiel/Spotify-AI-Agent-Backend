"""
Module: encryption.py
Description: Chiffrement industriel (AES, Fernet), gestion des clés, hashing, pour stockage sécurisé et transmission.
"""
from cryptography.fernet import Fernet
import hashlib
from typing import Optional

class EncryptionManager:
    @staticmethod
    def generate_key() -> bytes:
        return Fernet.generate_key()

    @staticmethod
    def encrypt(data: bytes, key: bytes) -> bytes:
        f = Fernet(key)
        return f.encrypt(data)

    @staticmethod
    def decrypt(token: bytes, key: bytes) -> bytes:
        f = Fernet(key)
        return f.decrypt(token)

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return hashlib.sha256(password.encode()).hexdigest() == hashed

# Exemples d'utilisation
# key = EncryptionManager.generate_key()
# token = EncryptionManager.encrypt(b"secret", key)
# EncryptionManager.decrypt(token, key)
# EncryptionManager.hash_password("mypassword")
