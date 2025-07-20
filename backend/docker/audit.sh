#!/bin/bash
# audit.sh – Spotify AI Agent
# --------------------------
# Security- und Compliance-Audit für alle Images & Container
# Rollen: Security Specialist, Lead Dev, Architecte IA

set -e

# Trivy Scan (Images)
if ! command -v trivy &> /dev/null; then
  echo "[ERROR] Trivy nicht installiert. Bitte installieren: https://aquasecurity.github.io/trivy/"
  exit 1
fi

for img in $(docker images --format '{{.Repository}}:{{.Tag}}'); do
  echo "[INFO] Scanne Image: $img"
  trivy image --severity CRITICAL,HIGH $img
done

# Container Security-Check
for c in $(docker ps --format '{{.Names}}'); do
  echo "[INFO] Prüfe laufenden Container: $c"
  docker exec "$c" sh -c 'id && whoami && env | grep -i secret && env | grep -i password' || true
  # Weitere Checks: offene Ports, Prozesse, Dateirechte
  docker exec "$c" sh -c 'netstat -tulpen || ss -tulpen'
  docker exec "$c" sh -c 'ps aux'
  docker exec "$c" sh -c 'ls -l /app || true'
  echo "---"
  sleep 1
  done

echo "[OK] Audit abgeschlossen. Siehe Ergebnisse oben."
