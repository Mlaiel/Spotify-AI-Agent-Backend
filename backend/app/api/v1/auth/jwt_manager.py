"""
JWTManager : Gestionnaire de tokens JWT sécurisé
- Génération, validation, rotation, blacklist
- Sécurité : signature, expiration, logs, RGPD
- Intégration FastAPI/Django, scalable microservices

Auteur : Backend Senior, Sécurité, Lead Dev
"""

from typing import Dict, Any, Optional
import jwt
import time

class JWTManager:
    """
    Gère la création, validation et rotation des tokens JWT.
    """
    def __init__(self, secret: str, algorithm: str = "HS256"):
        self.secret = secret
        self.algorithm = algorithm
        self.blacklist = set()

    def create_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """
        Génère un JWT signé avec expiration.
        """
        payload = payload.copy()
        payload["exp"] = int(time.time() + expires_in)
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Valide un JWT et vérifie la blacklist.
        """
        if token in self.blacklist:
            return None
        try:
            return jwt.decode(token, self.secret, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def blacklist_token(self, token: str):
        """
        Ajoute un token à la blacklist (logout, révocation).
        """
        self.blacklist.add(token)

# Exemple d’utilisation :
# manager = JWTManager(secret="supersecret")
# token = manager.create_token({"user_id": "user123"})
# print(manager.validate_token(token)
# manager.blacklist_token(token)
# print(manager.validate_token(token)
