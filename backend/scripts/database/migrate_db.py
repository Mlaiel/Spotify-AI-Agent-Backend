"""
migrate_db.py – Spotify AI Agent
-------------------------------
Führt alle Datenbankmigrationen (Postgres/Alembic, MongoDB, ElasticSearch) aus.
Rollen: DBA/Data Engineer, Backend Senior, Lead Dev
"""

import os
import logging
import subprocess

logging.basicConfig(level=logging.INFO)

# Alembic Migration (Postgres)
try:
    subprocess.run(["alembic", "upgrade", "head"], check=True)
    logging.info("[OK] Alembic Migration abgeschlossen.")
except Exception as e:
    logging.warning(f"[WARN] Alembic: {e}")

# MongoDB Migration (Beispiel)
try:
    if os.path.exists("../../migrations/mongodb/migrate.js"):
        subprocess.run(["mongo", os.environ["MONGO_URI"], "../../migrations/mongodb/migrate.js"], check=True)
        logging.info("[OK] MongoDB Migration abgeschlossen.")
except Exception as e:
    logging.warning(f"[WARN] MongoDB: {e}")

# ElasticSearch Migration (Beispiel)
try:
    if os.path.exists("../../migrations/elasticsearch/migrate.sh"):
        subprocess.run(["bash", "../../migrations/elasticsearch/migrate.sh"], check=True)
        logging.info("[OK] ElasticSearch Migration abgeschlossen.")
except Exception as e:
    logging.warning(f"[WARN] ElasticSearch: {e}")
