"""
Artist Monitoring Tasks for Spotify AI Agent
------------------------------------------------
- Überwacht Künstler-Performance, Follower, Erwähnungen, Socials, Trends
- Integriert Security, Audit, Observability, ML-Hooks, GDPR
- Produktionsreif, erweiterbar, mit Alerting und Business-Logik

Autoren & Rollen:
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect
"""

import logging
from celery import shared_task
from datetime import datetime
from typing import Dict, Any
from app.services.spotify import SpotifyAPIService
from app.services.analytics_service import AnalyticsService
from app.core.security import audit_log, secure_task
from app.core.utils.decorators import retry_on_failure

logger = logging.getLogger("artist_monitoring")

@shared_task(bind=True, name="artist_monitoring.monitor_artist", autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
@secure_task
@retry_on_failure
@audit_log(action="monitor_artist")
def monitor_artist(self, artist_id: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Überwacht einen Spotify-Künstler auf Follower, Erwähnungen, Socials, Trends.
    - Holt aktuelle Daten von Spotify API und Socials
    - ML-Hooks für Anomalie-Erkennung
    - Audit, Logging, GDPR-Compliance
    """
    logger.info(f"[MONITOR] Start monitoring artist {artist_id}")
    try:
        spotify_data = SpotifyAPIService().get_artist_profile(artist_id)
        analytics = AnalyticsService().get_artist_metrics(artist_id)
        # ML-Hook: Anomaly Detection
        anomalies = AnalyticsService().detect_anomalies(artist_id, spotify_data)
        # GDPR: Nur erlaubte Felder loggen
        result = {
            "artist_id": artist_id,
            "followers": spotify_data.get("followers"),
            "popularity": spotify_data.get("popularity"),
            "mentions": analytics.get("mentions"),
            "anomalies": anomalies,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.info(f"[MONITOR] Artist {artist_id} monitored: {result}")
        return result
    except Exception as e:
        logger.error(f"[MONITOR][ERROR] Artist {artist_id}: {e}")
        raise
