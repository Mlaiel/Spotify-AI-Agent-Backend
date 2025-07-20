import logging
from fastapi import WebSocket, WebSocketDisconnect, status
from typing import Dict, Any
from .connection_manager import ConnectionManager
from .middleware.auth_jwt import require_jwt
from .monitoring.metrics import ws_connections_total, ws_disconnections_total, ws_messages_total, ws_errors_total

class MusicStreamingHandler:
    """
    Gestionnaire de streaming audio temps réel via WebSocket (pré-écoute, monitoring, sécurité).
    """
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.logger = logging.getLogger("MusicStreamingHandler")

    async def handle(self, websocket: WebSocket, room: str, user_id: str):
        payload = require_jwt(websocket)
        ws_connections_total.inc()
        await self.manager.connect(websocket, room)
        try:
            while True:
                data = await websocket.receive_bytes()
                ws_messages_total.inc()
                # Ici, on pourrait ajouter une validation IA (ex: anti-piratage, fingerprinting)
                await self.manager.broadcast(
                    message=data,  # Pour l'audio, utiliser send_bytes côté manager si besoin
                    room=room)
                )
                self.logger.info(f"Chunk audio diffusé (room={room}, user={user_id}, size={len(data)})")
        except WebSocketDisconnect:
            self.manager.disconnect(websocket, room)
            ws_disconnections_total.inc()
            self.logger.info(f"Déconnexion streaming audio (room={room}, user={user_id})")
        except Exception as e:
            ws_errors_total.inc()
            self.logger.error(f"Erreur WebSocket streaming audio: {e}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

# Exemple d'intégration FastAPI
# from fastapi import APIRouter
# router = APIRouter()
# manager = ConnectionManager()
# music_handler = MusicStreamingHandler(manager)
# @router.websocket("/ws/music/{room}/{user_id}")
# async def websocket_music(websocket: WebSocket, room: str, user_id: str):
#     await music_handler.handle(websocket, room, user_id)
