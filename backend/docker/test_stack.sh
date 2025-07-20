#!/bin/bash
# test_stack.sh – Spotify AI Agent
# -------------------------------
# Testet, ob alle Kernservices im Docker-Stack laufen und erreichbar sind
# Rollen: DevOps, Lead Dev, Security Specialist

set -e

SERVICES=(backend celery-worker celery-beat redis postgres nginx)

for s in "${SERVICES[@]}"; do
  if docker ps --format '{{.Names}}' | grep -q "$s"; then
    echo "[OK] $s läuft."
  else
    echo "[ERROR] $s NICHT gefunden!" >&2
    exit 1
  fi
done

# Health-Checks
./healthcheck.sh

echo "[OK] Stack-Test abgeschlossen."
