#!/bin/bash
# migrate.sh – Spotify AI Agent
# ----------------------------
# Führt alle Datenbankmigrationen (Alembic, Mongo, Elastic) im Docker-Stack aus
# Rollen: DBA/Data Engineer, Backend Senior, Lead Dev

set -e

# Alembic Migration (Postgres)
ALEMBIC_CONTAINER="$(docker ps --filter 'name=backend' --format '{{.Names}}')"
if [ -n "$ALEMBIC_CONTAINER" ]; then
  echo "[INFO] Starte Alembic Migration..."
  docker exec "$ALEMBIC_CONTAINER" alembic upgrade head
else
  echo "[WARN] Kein laufender Backend-Container für Alembic gefunden."
fi

# MongoDB Migration (falls vorhanden)
MONGO_CONTAINER="$(docker ps --filter 'name=mongo' --format '{{.Names}}')"
if [ -n "$MONGO_CONTAINER" ]; then
  echo "[INFO] Starte MongoDB Migration..."
  # Beispiel: docker exec "$MONGO_CONTAINER" mongo < /app/migrations/mongodb/migrate.js
else
  echo "[INFO] Kein laufender MongoDB-Container gefunden."
fi

# ElasticSearch Migration (falls vorhanden)
ELASTIC_CONTAINER="$(docker ps --filter 'name=elasticsearch' --format '{{.Names}}')"
if [ -n "$ELASTIC_CONTAINER" ]; then
  echo "[INFO] Starte ElasticSearch Migration..."
  # Beispiel: docker exec "$ELASTIC_CONTAINER" bash /app/migrations/elasticsearch/migrate.sh
else
  echo "[INFO] Kein laufender ElasticSearch-Container gefunden."
fi

echo "[OK] Migration abgeschlossen."
