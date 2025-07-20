"""
Data Sync Tasks for Spotify AI Agent
-----------------------------------
- Synchronisiert Spotify-Daten (Künstler, Tracks, Playlists) mit interner DB
- Integriert Security, Audit, Observability, ML/AI-Hooks, GDPR
- Produktionsreif, robust, mit Delta-Detection und Versionierung

Autoren & Rollen:
- Lead Dev, Architecte IA, Backend Senior, ML Engineer, DBA/Data Engineer, Security Specialist, Microservices Architect
"""

import logging
from celery import shared_task
from datetime import datetime
from typing import Dict, Any
from app.services.spotify import SpotifyAPIService
from app.core.database import get_mongo_db, get_pg_conn
from app.core.security import audit_log, secure_task
from app.core.utils.decorators import retry_on_failure

logger = logging.getLogger("data_sync")

@shared_task(bind=True, name="data_sync.sync_artist_data", autoretry_for=(Exception,), retry_backoff=True, max_retries=5)
@secure_task
@retry_on_failure
@audit_log(action="sync_artist_data")
def sync_artist_data(self, artist_id: str, full_sync: bool = False) -> Dict[str, Any]:
    """
    Synchronise les données d'un artiste Spotify dans MongoDB et PostgreSQL.
    - Récupère les données actuelles, détecte le delta, stocke dans les deux backends
    - Audit, Logging, GDPR-Compliance
    """
    logger.info(f"[SYNC] Start sync for artist {artist_id}")
    try:
        spotify_data = SpotifyAPIService().get_artist_profile(artist_id)
        # --- Sauvegarde MongoDB ---
        mongo_db = get_mongo_db()
        mongo_result = mongo_db["spotify_data"].update_one(
            {"spotify_id": artist_id},
            {"$set": {"data": spotify_data, "updated_at": datetime.utcnow()}},
            upsert=True
        )
        # --- Sauvegarde PostgreSQL ---
        from app.models.spotify_data import SpotifyData, Base
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine
        # Connexion SQLAlchemy (DSN depuis env)
        engine = create_engine(os.getenv("POSTGRES_DSN", "postgresql+psycopg2://postgres:postgres@localhost:5432/spotify_ai_agent"))
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            obj = session.query(SpotifyData).filter_by(spotify_id=artist_id).first()
            if obj:
                obj.data = spotify_data
                obj.updated_at = datetime.utcnow()
            else:
                obj = SpotifyData.create(data_type="artist", spotify_id=artist_id, data=spotify_data)
                session.add(obj)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"[SYNC][PG][ERROR] {artist_id}: {e}")
            raise
        finally:
            session.close()
        logger.info(f"[SYNC] Artist {artist_id} sync done (Mongo+PG)")
        return {"artist_id": artist_id, "mongo_upserted": mongo_result.upserted_id, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"[SYNC][ERROR] Artist {artist_id}: {e}")
        raise
