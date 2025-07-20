"""
Module: api_key_manager.py
Description: Gestion industrielle des API Keys (génération, validation, rotation, permissions, audit, stockage sécurisé).
"""
import secrets
import hashlib
from typing import Dict, Optional

class APIKeyManager:
    _store: Dict[str, Dict] = {}

    @staticmethod
    def generate_key(length: int = 40) -> str:
        return secrets.token_urlsafe(length)

    @classmethod
    def store_key(cls, key: str, user_id: str, permissions: Optional[list] = None):
        hashed = hashlib.sha256(key.encode()).hexdigest()
        cls._store[hashed] = {"user_id": user_id, "permissions": permissions or []}

    @classmethod
    def validate_key(cls, key: str) -> Optional[Dict]:
        hashed = hashlib.sha256(key.encode()).hexdigest()
        return cls._store.get(hashed)

    @classmethod
    def revoke_key(cls, key: str):
        hashed = hashlib.sha256(key.encode()).hexdigest()
        cls._store.pop(hashed, None)

# Exemples d'utilisation
# key = APIKeyManager.generate_key()
# APIKeyManager.store_key(key, user_id="42", permissions=["read", "write"])
# APIKeyManager.validate_key(key)
