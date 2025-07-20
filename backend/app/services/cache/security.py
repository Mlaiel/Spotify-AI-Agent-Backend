import logging
import base64
from cryptography.fernet import Fernet
import os

logger = logging.getLogger("cache_security")

class CacheSecurity:
    """
    Sécurité avancée du cache : chiffrement, audit, RBAC, validation entrées/sorties, logs enrichis.
    Utilisable pour IA, analytics, Spotify, scoring, etc.
    """
    def __init__(self, key: str = None, rbac: dict = None):
        self.key = key or os.getenv("CACHE_ENCRYPTION_KEY") or Fernet.generate_key()
        self.fernet = Fernet(self.key)
        self.rbac = rbac or {"default": ["read", "write", "invalidate"]}
    def encrypt(self, value, user: str = "default"):
        self._audit("encrypt", user)
        if not self._check_permission(user, "write"):
            logger.error(f"RBAC: accès refusé pour {user} (write)")
            raise PermissionError("Accès refusé")
        if isinstance(value, str):
            value = value.encode()
        encrypted = self.fernet.encrypt(value)
        logger.debug("Valeur chiffrée pour le cache.")
        return base64.b64encode(encrypted).decode()
    def decrypt(self, value, user: str = "default"):
        self._audit("decrypt", user)
        if not self._check_permission(user, "read"):
            logger.error(f"RBAC: accès refusé pour {user} (read)")
            raise PermissionError("Accès refusé")
        try:
            decrypted = self.fernet.decrypt(base64.b64decode(value))
            logger.debug("Valeur déchiffrée du cache.")
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Erreur de déchiffrement: {e}")
            return None
    def _audit(self, action: str, user: str):
        logger.info(f"[AUDIT] {action} par {user}")
    def _check_permission(self, user: str, action: str) -> bool:
        return action in self.rbac.get(user, self.rbac["default"])
    def validate(self, value) -> bool:
        # Validation métier (exemple: JSON, schéma, etc.)
        return value is not None
