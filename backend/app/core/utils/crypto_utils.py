"""
Module: crypto_utils.py
Description: Utilitaires cryptographiques industriels (hash, signature, random, base64, HMAC, vérification).
"""
import hashlib
import hmac
import base64
import secrets
from typing import Optional

def hash_sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()

def hmac_sign(data: str, key: str) -> str:
    return hmac.new(key.encode(), data.encode(), hashlib.sha256).hexdigest()

def verify_hmac(data: str, key: str, signature: str) -> bool:
    return hmac.compare_digest(hmac_sign(data, key), signature)

def random_token(length: int = 32) -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode("utf-8").rstrip("=")

# Exemples d'utilisation
# hash_sha256("secret")
# hmac_sign("data", "key")
# verify_hmac("data", "key", "signature")
# random_token(16)
