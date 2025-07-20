import logging
from typing import Dict, Set
from fastapi import WebSocket

class ConnectionManager:
    """
    Gestionnaire avancé de connexions WebSocket : multi-clients, rooms, audit, sécurité.
    """
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.rooms: Dict[str, Set[WebSocket] = {}
        self.logger = logging.getLogger("ConnectionManager")

    async def connect(self, websocket: WebSocket, room: str = None):
        await websocket.accept()
        self.active_connections.add(websocket)
        if room:
            self.rooms.setdefault(room, set().add(websocket)
        self.logger.info(f"WebSocket connecté (room={room})")

    def disconnect(self, websocket: WebSocket, room: str = None):
        self.active_connections.discard(websocket)
        if room and room in self.rooms:
            self.rooms[room].discard(websocket)
        self.logger.info(f"WebSocket déconnecté (room={room})")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str, room: str = None):
        targets = self.rooms[room] if room and room in self.rooms else self.active_connections
        for connection in targets:
            await connection.send_text(message)
