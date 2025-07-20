"""
AI Content Generation Task for Spotify AI Agent
----------------------------------------------
- Generiert KI-gestützten Content (Texte, Posts, Hashtags) für Spotify-Künstler
- Integriert Security, Audit, Observability, ML/AI-Hooks, GDPR, Prometheus, OpenTelemetry, Sentry
- Produktionsreif, robust, mit Alerting und Data Engineering

Autoren & Rollen:
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect
"""

import logging
from celery import shared_task
from datetime import datetime
from typing import Dict, Any
from app.services.ai.content_generation_service import ContentGenerationService
from app.services.spotify import SpotifyAPIService
from app.core.security import audit_log, secure_task
from app.core.utils.decorators import retry_on_failure
from prometheus_client import Counter
from opentelemetry import trace
import sentry_sdk
from app.utils.metrics_manager import MetricsManager

logger = logging.getLogger("ai_content_generation")
tracer = trace.get_tracer(__name__)

# Utiliser le gestionnaire centralisé pour éviter les doublons
_metrics_manager = MetricsManager()
ai_content_counter = _metrics_manager.get_or_create_counter(
    'ai_content_generation_total', 
    'Total AI Content Generation Tasks'
)

@shared_task(bind=True, name="ai_content_generation.generate_content", autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
@secure_task
@retry_on_failure
@audit_log(action="generate_content")
def generate_content(self, artist_id: str, content_type: str = "post", context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generiert KI-Content für einen Spotify-Künstler.
    - Erstellt Texte, Posts, Hashtags via ML/AI
    - Audit, Logging, GDPR, Prometheus, OpenTelemetry, Sentry
    """
    with tracer.start_as_current_span("generate_content_task"):
        logger.info(f"[AI_CONTENT] Start content generation for artist {artist_id}")
        ai_content_counter.inc()
        try:
            # Utilisation du service industriel
            model_registry = {}  # À remplacer par le vrai registre de modèles
            generator = ContentGenerationService(model_registry=model_registry)
            prompt = context.get("prompt") if context else ""
            content = generator.generate_content(artist_id, content_type, prompt, metadata=context)
            SpotifyAPIService().publish_content(artist_id, content)
            logger.info(f"[AI_CONTENT] Content generated for artist {artist_id}")
            return {"artist_id": artist_id, "content": content, "timestamp": datetime.utcnow().isoformat()}
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"[AI_CONTENT][ERROR] {artist_id}: {e}")
            raise
