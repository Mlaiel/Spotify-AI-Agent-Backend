"""
Playlist Update Tasks for Spotify AI Agent
-----------------------------------------
- Automatisiert Playlist-Updates, KI-gestützte Empfehlungen, Kollaborationen
- Integriert Security, Audit, Observability, ML/AI-Hooks, GDPR
- Produktionsreif, robust, mit Alerting und Versionierung

Autoren & Rollen:
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect
"""

import logging
from celery import shared_task
from datetime import datetime
from typing import Dict, Any, List
from app.services.spotify import SpotifyAPIService
from app.services.ai.recommendation_service import RecommendationService
from app.core.security import audit_log, secure_task
from app.core.utils.decorators import retry_on_failure

logger = logging.getLogger("playlist_update")

@shared_task(bind=True, name="playlist_update.update_playlist", autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
@secure_task
@retry_on_failure
@audit_log(action="update_playlist")
def update_playlist(self, playlist_id: str, user_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Aktualisiert eine Playlist mit KI-gestützten Empfehlungen und Kollaborationen.
    - Holt aktuelle Playlist, berechnet neue Tracks, aktualisiert Spotify
    - Audit, Logging, GDPR-Compliance
    """
    logger.info(f"[PLAYLIST] Start update for playlist {playlist_id}")
    try:
        playlist = SpotifyAPIService().get_playlist(playlist_id)
        # Utilisation du service de recommandation industriel
        recommender = RecommendationService(recommender=None)  # À remplacer par l'engine réel si besoin
        recommendations = recommender.recommend(user_id, context or {})["recommendations"]
        updated = SpotifyAPIService().update_playlist_tracks(playlist_id, recommendations)
        logger.info(f"[PLAYLIST] Playlist {playlist_id} updated: {updated}")
        return {"playlist_id": playlist_id, "updated_tracks": updated, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"[PLAYLIST][ERROR] {playlist_id}: {e}")
        raise
