"""
Streaming Metrics Tasks for Spotify AI Agent
-------------------------------------------
- Aggregiert, analysiert und überwacht Streaming-Metriken in Echtzeit
- Integriert Security, Audit, Observability, ML/AI-Hooks, GDPR
- Produktionsreif, robust, mit Alerting, Monitoring und Data Engineering

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

logger = logging.getLogger("streaming_metrics")

@shared_task(bind=True, name="streaming_metrics.aggregate_metrics", autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
@secure_task
@retry_on_failure
@audit_log(action="aggregate_metrics")
def aggregate_metrics(self, artist_id: str, period: str = "24h") -> Dict[str, Any]:
    """
    Aggregiert und analysiert Streaming-Metriken für einen Künstler.
    - Holt Daten von Spotify, berechnet KPIs, speichert in Analytics DB
    - Audit, Logging, GDPR-Compliance
    """
    logger.info(f"[METRICS] Aggregating metrics for artist {artist_id} period {period}")
    try:
        metrics = SpotifyAPIService().get_streaming_metrics(artist_id, period)
        kpis = AnalyticsService().compute_kpis(metrics)
        AnalyticsService().store_metrics(artist_id, kpis)
        logger.info(f"[METRICS] Metrics for artist {artist_id} aggregated: {kpis}")
        return {"artist_id": artist_id, "kpis": kpis, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"[METRICS][ERROR] Artist {artist_id}: {e}")
        raise
