#!/bin/bash
# restore.sh – Spotify AI Agent
# ----------------------------
# Automatisiertes Restore für Postgres und Redis (prod/dev)
# Rollen: DBA/Data Engineer, Security Specialist, Lead Dev

set -e
BACKUP_DIR="$1"
if [ -z "$BACKUP_DIR" ]; then
  echo "Usage: $0 <backup-dir>"
  exit 1
fi

# Postgres Restore
PG_CONTAINER="$(docker ps --filter 'name=postgres' --format '{{.Names}}')"
if [ -n "$PG_CONTAINER" ] && [ -f "$BACKUP_DIR/postgres.sql" ]; then
  echo "[INFO] Restore Postgres..."
  cat "$BACKUP_DIR/postgres.sql" | docker exec -i "$PG_CONTAINER" psql -U $POSTGRES_USER
else
  echo "[WARN] Kein laufender Postgres-Container oder kein Backup gefunden."
fi

# Redis Restore
REDIS_CONTAINER="$(docker ps --filter 'name=redis' --format '{{.Names}}')"
if [ -n "$REDIS_CONTAINER" ] && [ -f "$BACKUP_DIR/redis.rdb" ]; then
  echo "[INFO] Restore Redis..."
  docker cp "$BACKUP_DIR/redis.rdb" "$REDIS_CONTAINER":/data/dump.rdb
  docker exec "$REDIS_CONTAINER" redis-cli shutdown nosave
  docker restart "$REDIS_CONTAINER"
else
  echo "[WARN] Kein laufender Redis-Container oder kein Backup gefunden."
fi

echo "[OK] Restore abgeschlossen."
