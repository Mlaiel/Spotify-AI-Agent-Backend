import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class SpotifyWebhookEvent(BaseModel):
    event_type: str = Field(..., description="Type d'événement Spotify (play, like, playlist_update, etc.)")
    user_id: str = Field(..., description="ID utilisateur Spotify")
    data: Dict[str, Any] = Field(..., description="Payload de l'événement")
    received_at: datetime = Field(default_factory=datetime.utcnow)

class SpotifyWebhook:
    """
    Gestion sécurisée des webhooks Spotify (écoutes, likes, playlists, audit, logs, sécurité).
    """
    def __init__(self):
        self.logger = logging.getLogger("SpotifyWebhook")
        self.events = []

    def handle_event(self, event: SpotifyWebhookEvent):
        # Ici, on pourrait valider la signature, stocker l'événement, déclencher des actions, etc.
        self.events.append(event)
        self.logger.info(f"Webhook Spotify reçu: {event.event_type} pour user {event.user_id}")
        return {"status": "ok", "received": event.event_type}

    def get_events(self, user_id: str = None):
        if user_id:
            return [e for e in self.events if e.user_id == user_id]
        return self.events
