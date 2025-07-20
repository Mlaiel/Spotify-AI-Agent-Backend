#!/bin/bash
# backup_db.sh – Spotify AI Agent
# ------------------------------
# Erstellt ein vollständiges Backup aller Datenbanken (Postgres, MongoDB, Redis)
# Rollen: DBA/Data Engineer, Security Specialist, Lead Dev

set -e
BACKUP_DIR="/backups/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BACKUP_DIR"

# Postgres Backup
if command -v pg_dumpall &> /dev/null; then
  echo "[INFO] Backup Postgres..."
  pg_dumpall -U "$POSTGRES_USER" > "$BACKUP_DIR/postgres.sql"
else
  echo "[WARN] pg_dumpall nicht gefunden."
fi

# MongoDB Backup
if command -v mongodump &> /dev/null; then
  echo "[INFO] Backup MongoDB..."
  mongodump --out "$BACKUP_DIR/mongo"
else
  echo "[INFO] mongodump nicht gefunden."
fi

# Redis Backup
if command -v redis-cli &> /dev/null; then
  echo "[INFO] Backup Redis..."
  redis-cli save
  cp /var/lib/redis/dump.rdb "$BACKUP_DIR/redis.rdb"
else
  echo "[INFO] redis-cli nicht gefunden."
fi

echo "[OK] Backup abgeschlossen: $BACKUP_DIR"
