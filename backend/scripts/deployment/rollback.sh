#!/bin/bash
# rollback.sh – Spotify AI Agent Deployment
# ----------------------------------------
# Automatisiertes, sicheres Rollback auf das letzte Backup bei Fehlern im Deployment.
# Rollen: Lead Dev, Architecte IA, Security Specialist, DBA/Data Engineer

set -e

# Letztes Backup finden
LAST_BACKUP=$(ls -td /deploy_backups/* | head -1)
if [ -z "$LAST_BACKUP" ]; then
  echo "[ERROR] Kein Backup gefunden! Rollback abgebrochen." >&2
  exit 1
fi

echo "[INFO] Rollback auf Backup: $LAST_BACKUP"

# Restore Datenbanken
if [ -d "../../scripts/database" ]; then
  bash ../../scripts/database/restore_db.sh "$LAST_BACKUP"
fi

# Restore Artefakte
cp -r "$LAST_BACKUP/app" ../../app
cp -r "$LAST_BACKUP/config" ../../config
cp -r "$LAST_BACKUP/ml" ../../ml
cp -r "$LAST_BACKUP/docker" ../../docker
cp -r "$LAST_BACKUP/logs" ../../logs 2>/dev/null || true

# Audit-Log
echo "[WARN] Rollback durchgeführt am $(date) von $USER auf $LAST_BACKUP" >> ../../logs/deployment.log

echo "[OK] Rollback abgeschlossen."
