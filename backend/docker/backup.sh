#!/bin/bash
# backup.sh – Spotify AI Agent
# ---------------------------
# Automatisiertes Backup für Postgres und Redis (prod/dev)
# Rollen: DBA/Data Engineer, Security Specialist, Lead Dev

set -e
BACKUP_DIR="/backups/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BACKUP_DIR"

# Postgres Backup
PG_CONTAINER="$(docker ps --filter 'name=postgres' --format '{{.Names}}')"
if [ -n "$PG_CONTAINER" ]; then
  echo "[INFO] Backup Postgres..."
  docker exec "$PG_CONTAINER" pg_dumpall -U $POSTGRES_USER > "$BACKUP_DIR/postgres.sql"
else
  echo "[WARN] Kein laufender Postgres-Container gefunden."
fi

# Redis Backup
REDIS_CONTAINER="$(docker ps --filter 'name=redis' --format '{{.Names}}')"
if [ -n "$REDIS_CONTAINER" ]; then
  echo "[INFO] Backup Redis..."
  docker exec "$REDIS_CONTAINER" redis-cli save
  docker cp "$REDIS_CONTAINER":/data/dump.rdb "$BACKUP_DIR/redis.rdb"
else
  echo "[WARN] Kein laufender Redis-Container gefunden."
fi

echo "[OK] Backup abgeschlossen: $BACKUP_DIR"
