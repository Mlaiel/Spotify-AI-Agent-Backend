import logging
from fastapi import WebSocket, status
from fastapi.exceptions import WebSocketException
import jwt
from typing import Optional

SECRET_KEY = "votre_cle_secrete"  # À externaliser dans la config/env
ALGORITHM = "HS256"

logger = logging.getLogger("AuthJWT")

def decode_jwt(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError as e:
        logger.warning(f"JWT invalide: {e}")
        return None

def require_jwt(websocket: WebSocket):
    async def dependency():
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Token JWT manquant")
        payload = decode_jwt(token)
        if not payload:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Token JWT invalide")
        return payload
    return dependency

# Exemple d'utilisation dans un endpoint FastAPI :
# from .middleware.auth_jwt import require_jwt
# @router.websocket("/ws/chat/{room}/{user_id}")
# async def websocket_chat(websocket: WebSocket, room: str, user_id: str):
#     await require_jwt(websocket)()
#     ...
