#!/bin/bash
# cleanup_logs.sh – Spotify AI Agent
# ----------------------------------
# Automatisiert, sichert und rotiert Logs (inkl. DSGVO/GDPR, Security, Compliance)
# Rollen: Lead Dev, Security Specialist, Architecte IA, Backend Senior

set -e

LOG_DIR="../../logs"
ARCHIVE_DIR="../../logs/archive/$(date +%Y-%m-%d)"
mkdir -p "$ARCHIVE_DIR"

# Log-Rotation & Archivierung
find "$LOG_DIR" -type f -name '*.log' -mtime +7 -exec mv {} "$ARCHIVE_DIR" \;

# DSGVO/GDPR: Lösche personenbezogene Daten aus alten Logs
find "$ARCHIVE_DIR" -type f -name '*.log' -exec sed -i -E 's/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}//g' {} \;

# Security: Setze restriktive Rechte
chmod -R 600 "$ARCHIVE_DIR"

# Compliance: Audit-Log
echo "[INFO] Log-Cleanup durchgeführt am $(date) von $USER" >> "$LOG_DIR/audit.log"

echo "[OK] Log-Cleanup abgeschlossen."
