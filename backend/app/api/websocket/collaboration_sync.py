import logging
from fastapi import WebSocket, WebSocketDisconnect, status
from typing import Dict, Any
from .connection_manager import ConnectionManager
from .middleware.auth_jwt import require_jwt
from .monitoring.metrics import ws_connections_total, ws_disconnections_total, ws_messages_total, ws_errors_total

class CollaborationSyncHandler:
    """
    Gestionnaire de synchronisation de collaboration en temps réel (présence, scoring, état partagé).
    """
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.logger = logging.getLogger("CollaborationSyncHandler")
        self.presence: Dict[str, str] = {}  # user_id -> room
        self.scores: Dict[str, int] = {}    # user_id -> score

    async def handle(self, websocket: WebSocket, room: str, user_id: str):
        payload = require_jwt(websocket)
        ws_connections_total.inc()
        await self.manager.connect(websocket, room)
        self.presence[user_id] = room
        try:
            while True:
                data = await websocket.receive_json()
                ws_messages_total.inc()
                event = data.get("event")
                payload = data.get("payload")
                if event == "presence_update":
                    self.presence[user_id] = room
                    await self.manager.broadcast(
                        message=f"[PRESENCE] {user_id} is online in {room}",
                        room=room)
                    )
                elif event == "score_update":
                    score = payload.get("score", 0)
                    self.scores[user_id] = score
                    await self.manager.broadcast(
                        message=f"[SCORE] {user_id} score: {score}",
                        room=room)
                    )
                # Ajout d'autres événements collaboratifs (ex: sync, lock, etc.)
        except WebSocketDisconnect:
            self.manager.disconnect(websocket, room)
            ws_disconnections_total.inc()
            self.presence.pop(user_id, None)
            self.logger.info(f"Déconnexion collaboration (room={room}, user={user_id})")
        except Exception as e:
            ws_errors_total.inc()
            self.logger.error(f"Erreur WebSocket collaboration: {e}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

# Exemple d'intégration FastAPI
# from fastapi import APIRouter
# router = APIRouter()
# manager = ConnectionManager()
# collab_handler = CollaborationSyncHandler(manager)
# @router.websocket("/ws/collab/{room}/{user_id}")
# async def websocket_collab(websocket: WebSocket, room: str, user_id: str):
#     await collab_handler.handle(websocket, room, user_id)
