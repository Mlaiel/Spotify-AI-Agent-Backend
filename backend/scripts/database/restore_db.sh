#!/bin/bash
# restore_db.sh – Spotify AI Agent
# -------------------------------
# Stellt ein vollständiges Backup aller Datenbanken (Postgres, MongoDB, Redis) wieder her
# Rollen: DBA/Data Engineer, Security Specialist, Lead Dev

set -e
BACKUP_DIR="$1"
if [ -z "$BACKUP_DIR" ]; then
  echo "Usage: $0 <backup-dir>"
  exit 1
fi

# Postgres Restore
if command -v psql &> /dev/null && [ -f "$BACKUP_DIR/postgres.sql" ]; then
  echo "[INFO] Restore Postgres..."
  psql -U "$POSTGRES_USER" -f "$BACKUP_DIR/postgres.sql"
else
  echo "[WARN] psql nicht gefunden oder kein Postgres-Backup vorhanden."
fi

# MongoDB Restore
if command -v mongorestore &> /dev/null && [ -d "$BACKUP_DIR/mongo" ]; then
  echo "[INFO] Restore MongoDB..."
  mongorestore "$BACKUP_DIR/mongo"
else
  echo "[INFO] mongorestore nicht gefunden oder kein Mongo-Backup vorhanden."
fi

# Redis Restore
if command -v redis-cli &> /dev/null && [ -f "$BACKUP_DIR/redis.rdb" ]; then
  echo "[INFO] Restore Redis..."
  cp "$BACKUP_DIR/redis.rdb" /var/lib/redis/dump.rdb
  redis-cli shutdown nosave
  systemctl restart redis || service redis restart
else
  echo "[INFO] redis-cli nicht gefunden oder kein Redis-Backup vorhanden."
fi

echo "[OK] Restore abgeschlossen."
