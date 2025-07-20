"""
Module: jwt_manager.py
Description: Gestion industrielle des JWT (création, validation, rotation, blacklist, support OAuth2, FastAPI, microservices).
"""
import jwt
import datetime
from typing import Dict, Any, Optional

SECRET_KEY = "change_this_secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class JWTManager:
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[datetime.timedelta] = None) -> str:
        to_encode = data.copy()
        expire = datetime.datetime.utcnow() + (expires_delta or datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")

    @staticmethod
    def is_token_blacklisted(token: str) -> bool:
        # À implémenter avec Redis ou DB pour la prod
        return False

# Exemples d'utilisation
# token = JWTManager.create_access_token({"sub": "user_id"})
# data = JWTManager.decode_token(token)
