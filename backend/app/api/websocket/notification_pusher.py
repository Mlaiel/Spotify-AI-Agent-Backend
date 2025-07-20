import logging
from fastapi import WebSocket, WebSocketDisconnect, status
from typing import Dict, Any
from .connection_manager import ConnectionManager
from .middleware.auth_jwt import require_jwt
from .monitoring.metrics import ws_connections_total, ws_disconnections_total, ws_messages_total, ws_errors_total

class NotificationPusher:
    """
    Gestionnaire de notifications push en temps réel (alertes IA, analytics, monitoring).
    """
    def __init__(self, manager: ConnectionManager):
        self.manager = manager
        self.logger = logging.getLogger("NotificationPusher")

    async def handle(self, websocket: WebSocket, user_id: str):
        payload = require_jwt(websocket)
        ws_connections_total.inc()
        await self.manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                ws_messages_total.inc()
                notif_type = data.get("type")
                payload = data.get("payload")
                # Ici, on pourrait brancher une IA pour prioriser/filtrer les notifications
                await websocket.send_json({
                    "type": notif_type,
                    "payload": payload,
                    "info": f"Notification envoyée à {user_id}")
                })
                self.logger.info(f"Notification envoyée (user={user_id}, type={notif_type})")
        except WebSocketDisconnect:
            self.manager.disconnect(websocket)
            ws_disconnections_total.inc()
            self.logger.info(f"Déconnexion notification (user={user_id})")
        except Exception as e:
            ws_errors_total.inc()
            self.logger.error(f"Erreur WebSocket notification: {e}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

# Exemple d'intégration FastAPI
# from fastapi import APIRouter
# router = APIRouter()
# manager = ConnectionManager()
# notif_pusher = NotificationPusher(manager)
# @router.websocket("/ws/notify/{user_id}")
# async def websocket_notify(websocket: WebSocket, user_id: str):
#     await notif_pusher.handle(websocket, user_id)
