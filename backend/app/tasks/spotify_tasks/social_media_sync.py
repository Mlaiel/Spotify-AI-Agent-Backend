"""
Social Media Sync Task for Spotify AI Agent
------------------------------------------
- Synchronisiert Social-Media-Daten (Twitter, Instagram, TikTok) mit Spotify-Künstlerprofilen
- Integriert Security, Audit, Observability, ML/AI-Hooks, GDPR, Prometheus, OpenTelemetry, Sentry
- Produktionsreif, robust, mit Alerting und Data Engineering

Autoren & Rollen:
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect
"""

import logging
from celery import shared_task
from datetime import datetime
from typing import Dict, Any
from app.services.social_media_service import SocialMediaService
from app.services.spotify import SpotifyAPIService
from app.core.security import audit_log, secure_task
from app.core.utils.decorators import retry_on_failure
from prometheus_client import Counter
from opentelemetry import trace
import sentry_sdk

from app.utils.metrics_manager import get_counter

logger = logging.getLogger("social_media_sync")
tracer = trace.get_tracer(__name__)
social_sync_counter = get_counter('social_media_sync_total', 'Total Social Media Sync Tasks')

@shared_task(bind=True, name="social_media_sync.sync_social_media", autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
@secure_task
@retry_on_failure
@audit_log(action="sync_social_media")
def sync_social_media(self, artist_id: str, platforms: list = None) -> Dict[str, Any]:
    """
    Synchronisiert Social-Media-Daten für einen Spotify-Künstler.
    - Holt aktuelle Social-Daten, mapped auf Spotify-Profil
    - Audit, Logging, GDPR, Prometheus, OpenTelemetry, Sentry
    """
    with tracer.start_as_current_span("sync_social_media_task"):
        logger.info(f"[SOCIAL] Start sync for artist {artist_id}")
        social_sync_counter.inc()
        try:
            data = SocialMediaService().fetch_artist_socials(artist_id, platforms)
            SpotifyAPIService().update_artist_socials(artist_id, data)
            logger.info(f"[SOCIAL] Artist {artist_id} social data synced: {data}")
            return {"artist_id": artist_id, "social_data": data, "timestamp": datetime.utcnow().isoformat()}
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"[SOCIAL][ERROR] Artist {artist_id}: {e}")
            raise
