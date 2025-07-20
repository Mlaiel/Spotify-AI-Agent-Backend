#!/bin/bash
# backup.sh – Spotify AI Agent Deployment
# --------------------------------------
# Erstellt ein vollständiges, versioniertes Backup aller kritischen Daten und Artefakte vor Deployment.
# Rollen: Lead Dev, DBA/Data Engineer, Security Specialist, Architecte IA

set -e
BACKUP_DIR="/deploy_backups/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$BACKUP_DIR"

# Backup Datenbanken
if [ -d "../../scripts/database" ]; then
  bash ../../scripts/database/backup_db.sh
  cp -r /backups/* "$BACKUP_DIR/"
fi

# Backup Artefakte (Code, Konfig, Modelle)
cp -r ../../app "$BACKUP_DIR/app"
cp -r ../../config "$BACKUP_DIR/config"
cp -r ../../ml "$BACKUP_DIR/ml"
cp -r ../../docker "$BACKUP_DIR/docker"

# Backup Logs
cp -r ../../logs "$BACKUP_DIR/logs" 2>/dev/null || true

# Hash für Integrität
find "$BACKUP_DIR" -type f -exec sha256sum {} \; > "$BACKUP_DIR/backup.sha256"

echo "[OK] Deployment-Backup abgeschlossen: $BACKUP_DIR"
