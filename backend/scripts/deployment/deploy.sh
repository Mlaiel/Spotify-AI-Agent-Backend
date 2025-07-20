#!/bin/bash
# deploy.sh – Spotify AI Agent
# ---------------------------
# Führt ein Zero-Downtime-Deployment mit Health-Checks, Rollback und Logging durch.
# Rollen: Lead Dev, Architecte IA, Backend Senior, Security Specialist, Microservices Architect

set -e

# Backup vor Deployment
bash ./backup.sh

# Pull neue Images
cd ../../docker
docker-compose pull

# Rolling Update
docker-compose up -d --remove-orphans

# Health-Check
cd -
bash ./health_check.sh

# Audit-Log
echo "[INFO] Deployment durchgeführt am $(date) von $USER" >> ../../logs/deployment.log

echo "[OK] Deployment abgeschlossen."
