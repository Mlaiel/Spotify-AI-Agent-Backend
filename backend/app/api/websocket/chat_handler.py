import logging
from fastapi import WebSocket, WebSocketDisconnect, status
from typing import Dict, Any
from .connection_manager import ConnectionManager
from .middleware.auth_jwt import require_jwt
from .monitoring.metrics import ws_connections_total, ws_disconnections_total, ws_messages_total, ws_errors_total

class ChatModerationAI:
    """
    IA de modération de chat (toxicity, spam, flood, etc.).
    Peut être branchée sur Hugging Face ou un modèle custom.
    """
    def moderate(self, message: str, user_id: str) -> bool:
        # TODO: Intégrer un vrai modèle ML (Hugging Face, etc.)
        # Ici, simple démo : refuse les messages contenant 'banned'
        return 'banned' not in message.lower()

class ChatHandler:
    """
    Gestionnaire de chat WebSocket : rooms, modération, logs, audit, hooks IA.
    """
    def __init__(self, manager: ConnectionManager, moderation_ai: ChatModerationAI = None):
        self.manager = manager
        self.logger = logging.getLogger("ChatHandler")
        self.moderation_ai = moderation_ai or ChatModerationAI()

    async def handle(self, websocket: WebSocket, room: str, user_id: str):
        payload = require_jwt(websocket)
        ws_connections_total.inc()
        await self.manager.connect(websocket, room)
        try:
            while True:
                data = await websocket.receive_json()
                ws_messages_total.inc()
                msg_type = data.get("type")
                content = data.get("content")
                if msg_type == "message":
                    if not self.moderation_ai.moderate(content, user_id):
                        await websocket.send_json({"type": "error", "detail": "Message non autorisé (modération)"})
                        self.logger.warning(f"Message refusé (user={user_id}) : {content}")
                        continue
                    await self.manager.broadcast(
                        message=f"[{user_id}] {content}",
                        room=room)
                    )
                    self.logger.info(f"Message diffusé (room={room}, user={user_id})")
                # Ajout d'autres types d'événements (ex: typing, join/leave, etc.)
        except WebSocketDisconnect:
            self.manager.disconnect(websocket, room)
            ws_disconnections_total.inc()
            self.logger.info(f"Déconnexion WebSocket (room={room}, user={user_id})")
        except Exception as e:
            ws_errors_total.inc()
            self.logger.error(f"Erreur WebSocket chat: {e}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

# Exemple d'intégration FastAPI
# from fastapi import APIRouter, Depends
# router = APIRouter()
# manager = ConnectionManager()
# chat_handler = ChatHandler(manager)
# @router.websocket("/ws/chat/{room}/{user_id}")
# async def websocket_chat(websocket: WebSocket, room: str, user_id: str):
#     await chat_handler.handle(websocket, room, user_id)
