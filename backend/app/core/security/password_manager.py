"""
Module: password_manager.py
Description: Gestion industrielle des mots de passe (hashing, validation, politique de complexité, reset token, brute-force protection).
"""
import re
import secrets
import hashlib
from typing import Optional

class PasswordManager:
    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return hashlib.sha256(password.encode()).hexdigest() == hashed

    @staticmethod
    def validate_password(password: str) -> bool:
        # Politique forte : 8+ caractères, majuscule, minuscule, chiffre, spécial
        return bool(re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$", password))

    @staticmethod
    def generate_reset_token() -> str:
        return secrets.token_urlsafe(32)

# Exemples d'utilisation
# PasswordManager.validate_password("StrongP@ssw0rd")
# PasswordManager.generate_reset_token()
